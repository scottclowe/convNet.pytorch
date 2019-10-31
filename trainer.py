import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.nn.utils import clip_grad_norm_
from utils.meters import AverageMeter, accuracy
from utils.mixup import MixUp
from random import sample


def _flatten_duplicates(inputs, target, batch_first=True):
    if batch_first:
        target = target.view(-1, 1).expand(-1, inputs.size(1))
    else:
        inputs = inputs.transpose(0, 1)
        target = target.view(1, -1).expand(inputs.size(0), -1)
    inputs = inputs.flatten(0, 1)
    target = target.flatten(0, 1)
    return inputs, target


def _mixup(mixup_modules, alpha, batch_size):
    mixup_layer = None
    if len(mixup_modules) > 0:
        for m in mixup_modules:
            m.reset()
        mixup_layer = sample(mixup_modules, 1)[0]
        mixup_layer.sample(alpha, batch_size)
    return mixup_layer


class Trainer(object):

    def __init__(self, model, criterion, optimizer=None,
                 device_ids=[0], device=torch.cuda, dtype=torch.float,
                 distributed=False, local_rank=-1, adapt_grad_norm=None,
                 mixup=None, loss_scale=1., grad_clip=-1, print_freq=100):
        self._model = model
        self.criterion = criterion
        self.epoch = 0
        self.training_steps = 0
        self.optimizer = optimizer
        self.device = device
        self.dtype = dtype
        self.local_rank = local_rank
        self.print_freq = print_freq
        self.grad_clip = grad_clip
        self.mixup = mixup
        self.grad_scale = None
        self.loss_scale = loss_scale
        self.adapt_grad_norm = adapt_grad_norm

        if distributed:
            self.model = nn.parallel.DistributedDataParallel(model,
                                                             device_ids=device_ids,
                                                             output_device=device_ids[0])
        elif device_ids and len(device_ids) > 1:
            self.model = nn.DataParallel(model, device_ids)
        else:
            self.model = model

    def _grad_norm(self, inputs_batch, target_batch, chunk_batch=1):
        self.model.zero_grad()
        for inputs, target in zip(inputs_batch.chunk(chunk_batch, dim=0),
                                  target_batch.chunk(chunk_batch, dim=0)):
            target = target.to(self.device)
            inputs = inputs.to(self.device, dtype=self.dtype)

            # compute output
            output = self.model(inputs)
            loss = self.criterion(output, target)

            if chunk_batch > 1:
                loss = loss / chunk_batch

            loss.backward()   # accumulate gradient
        grad = clip_grad_norm_(self.model.parameters(), float('inf'))
        return grad

    def _step(self, inputs_batch, target_batch, training=False, chunk_batch=1):
        outputs = []
        total_loss = 0

        if training:
            self.optimizer.zero_grad()
            self.optimizer.update(self.epoch, self.training_steps)

        for i, (inputs, target) in enumerate(zip(inputs_batch.chunk(chunk_batch, dim=0),
                                                 target_batch.chunk(chunk_batch, dim=0))):
            target = target.to(self.device)
            inputs = inputs.to(self.device, dtype=self.dtype)

            mixup = None
            if training:
                self.optimizer.pre_forward()
                if self.mixup is not None:
                    input_mixup = MixUp()
                    mixup_modules = [input_mixup]  # input mixup
                    mixup_modules += [m for m in self.model.modules()
                                      if isinstance(m, MixUp)]
                    mixup = _mixup(mixup_modules, self.mixup, inputs.size(0))
                    inputs = input_mixup(inputs)

            # compute output
            output = self.model(inputs)
            if mixup is not None:
                target = mixup.mix_target(target, output.size(-1))
            loss = self.criterion(output, target)
            grad = None

            if chunk_batch > 1:
                loss = loss / chunk_batch

            if isinstance(output, list) or isinstance(output, tuple):
                output = output[0]

            outputs.append(output.detach())
            total_loss += float(loss)

            if training:
                if i == 0:
                    self.optimizer.pre_backward()
                if self.grad_scale is not None:
                    loss = loss * self.grad_scale
                if self.loss_scale is not None:
                    loss = loss * self.loss_scale
                loss.backward()   # accumulate gradient

        if training:  # post gradient accumulation
            if self.loss_scale is not None:
                for p in self.model.parameters():
                    if p.grad is None:
                        continue
                    p.grad.data.div_(self.loss_scale)

            if self.grad_clip > 0:
                grad = clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()  # SGD step
            self.training_steps += 1

        outputs = torch.cat(outputs, dim=0)
        return outputs, total_loss, grad

    def forward(self, data_loader, num_steps=None, training=False, duplicates=1, chunk_batch=1):
        meters = {name: AverageMeter()
                  for name in ['step', 'data', 'loss', 'prec1', 'prec5']}
        if training and self.grad_clip > 0:
            meters['grad'] = AverageMeter()

        batch_first = True
        if training and isinstance(self.model, nn.DataParallel) or chunk_batch > 1:
            batch_first = False

        def meter_results(meters):
            results = {name: meter.avg for name, meter in meters.items()}
            results['error1'] = 100. - results['prec1']
            results['error5'] = 100. - results['prec5']
            return results

        end = time.time()

        for i, (inputs, target) in enumerate(data_loader):
            if training and duplicates > 1 and self.adapt_grad_norm is not None \
                    and i % self.adapt_grad_norm == 0:
                grad_mean = 0
                num = inputs.size(1)
                for j in range(num):
                    grad_mean += float(self._grad_norm(inputs.select(1, j), target))
                grad_mean /= num
                grad_all = float(self._grad_norm(
                    *_flatten_duplicates(inputs, target, batch_first)))
                self.grad_scale = grad_mean / grad_all
                logging.info('New loss scale: %s', self.grad_scale)

            # measure data loading time
            meters['data'].update(time.time() - end)
            if duplicates > 1:  # multiple versions for each sample (dim 1)
                inputs, target = _flatten_duplicates(
                    inputs, target, batch_first)

            output, loss, grad = self._step(inputs, target,
                                            training=training,
                                            chunk_batch=chunk_batch)

            current_lr = self.optimizer.optimizer.param_groups[0]['lr']

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            meters['loss'].update(float(loss), inputs.size(0))
            meters['prec1'].update(float(prec1), inputs.size(0))
            meters['prec5'].update(float(prec5), inputs.size(0))
            if grad is not None:
                meters['grad'].update(float(grad), inputs.size(0))

            # measure elapsed time
            meters['step'].update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:
                report = str(
                    '{phase} - Epoch {0:3d} [{1:3d}/{2:3d}]   '
                    'LR {:.3E}   '
                    'Time {meters[step].val:.3f} ({meters[step].avg:.3f})   '
                    'Data {meters[data].val:.3f} ({meters[data].avg:.3f})   '
                    'Loss {meters[loss].val:.4f} ({meters[loss].avg:.4f})   '
                    'Prec@1 {meters[prec1].val:7.3f} ({meters[prec1].avg:7.3f})   '
                    'Prec@5 {meters[prec5].val:7.3f} ({meters[prec5].avg:7.3f})   '
                    .format(
                        self.epoch + 1, i, len(data_loader),
                        current_lr,
                        phase='TRAINING  ' if training else 'EVALUATING',
                        meters=meters
                    )
                )
                if 'grad' in meters.keys():
                    report += 'Grad {meters[grad].val:.3f} ({meters[grad].avg:.3f})'\
                        .format(meters=meters)
                logging.info(report)

            if num_steps is not None and i >= num_steps:
                break

        return meter_results(meters)

    def train(self, data_loader, duplicates=1, chunk_batch=1):
        # switch to train mode
        self.model.train()

        return self.forward(data_loader, duplicates=duplicates, training=True, chunk_batch=chunk_batch)

    def validate(self, data_loader, duplicates=1):
        # switch to evaluate mode
        self.model.eval()
        with torch.no_grad():
            return self.forward(data_loader, duplicates=duplicates, training=False)
