import argparse
import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import models
import torch.distributed as dist
from data import DataRegime
from utils.log import setup_logging, ResultsLog, save_checkpoint
from utils.optim import OptimRegime
from utils.cross_entropy import CrossEntropyLoss
from utils.misc import torch_dtypes
from utils.param_filter import FilterModules, is_bn
from datetime import datetime
from ast import literal_eval
from trainer import Trainer

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def main(
        results_dir='./results',
        save='',
        datasets_dir='~/Datasets',
        dataset='imagenet',
        model='alexnet',
        input_size=None,
        model_config='',
        dtype='float',
        device='cuda',
        world_size=-1,
        local_rank=-1,
        dist_init='env://',
        workers=8,
        epochs=90,
        start_epoch=-1,
        batch_size=256,
        eval_batch_size=-1,
        optimizer='SGD',
        label_smoothing=0,
        mixup=None,
        duplicates=1,
        chunk_batch=1,
        cutout=False,
        autoaugment=False,
        grad_clip=-1,
        loss_scale=1,
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4,
        print_freq=10,
        adapt_grad_norm=None,
        resume='',
        evaluate='',
        seed=123,
        subsplit_str='',
        augval=False,
    ):

    best_prec1 = 0
    dtype_str = dtype
    dtype = torch_dtypes.get(dtype_str)
    torch.manual_seed(seed)
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if evaluate:
        results_dir = '/tmp'
    if save is '':
        save = time_stamp
    save_path = os.path.join(results_dir, save)

    distributed = local_rank >= 0 or world_size > 1

    if distributed:
        dist.init_process_group(backend=dist_backend, init_method=dist_init,
                                world_size=world_size, rank=local_rank)
        local_rank = dist.get_rank()
        world_size = dist.get_world_size()
        if dist_backend == 'mpi':
            # If using MPI, select all visible devices
            device_ids = list(range(torch.cuda.device_count()))
        else:
            device_ids = [local_rank]

    if not os.path.exists(save_path) and not (distributed and local_rank > 0):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'),
                  resume=resume is not '',
                  dummy=distributed and local_rank > 0)

    results_path = os.path.join(save_path, 'results')
    results = ResultsLog(
        results_path, title='Training Results - %s' % save)

    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)
    logging.info("creating model %s", model)

    if 'cuda' in device and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.set_device(device_ids[0])
        cudnn.benchmark = True
    else:
        device_ids = None

    # create model
    model = models.__dict__[model]
    model_config = {'dataset': dataset}

    if model_config is not '':
        model_config = dict(model_config, **literal_eval(model_config))

    model = model(**model_config)
    logging.info("created model with configuration: %s", model_config)
    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    # optionally resume from a checkpoint
    if evaluate:
        if not os.path.isfile(evaluate):
            parser.error('invalid checkpoint: {}'.format(evaluate))
        checkpoint = torch.load(evaluate)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info("loaded checkpoint '%s' (epoch %s)",
                     evaluate, checkpoint['epoch'])
    elif resume:
        checkpoint_file = resume
        if os.path.isdir(checkpoint_file):
            results.load(os.path.join(checkpoint_file, 'results.csv'))
            checkpoint_file = os.path.join(
                checkpoint_file, 'model_best.pth.tar')
        if os.path.isfile(checkpoint_file):
            logging.info("loading checkpoint '%s'", resume)
            checkpoint = torch.load(checkpoint_file)
            if start_epoch < 0:  # not explicitly set
                start_epoch = checkpoint['epoch'] - 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         checkpoint_file, checkpoint['epoch'])
        else:
            logging.error("no checkpoint found at '%s'", resume)

    # define loss function (criterion) and optimizer
    loss_params = {}
    if label_smoothing > 0:
        loss_params['smooth_eps'] = label_smoothing
    criterion = getattr(model, 'criterion', CrossEntropyLoss)(**loss_params)
    criterion.to(device, dtype)
    model.to(device, dtype)

    # Batch-norm should always be done in float
    if 'half' in dtype_str:
        FilterModules(model, module=is_bn).to(dtype=torch.float)

    # optimizer configuration
    optim_regime = getattr(model, 'regime', [{'epoch': 0,
                                              'optimizer': optimizer,
                                              'lr': lr,
                                              'momentum': momentum,
                                              'weight_decay': weight_decay}])

    optimizer = optim_regime if isinstance(optim_regime, OptimRegime) \
        else OptimRegime(model, optim_regime, use_float_copy='half' in dtype_str)

    trainer = Trainer(model, criterion, optimizer,
                      device_ids=device_ids, device=device, dtype=dtype,
                      distributed=distributed, local_rank=local_rank, mixup=mixup, loss_scale=loss_scale,
                      grad_clip=grad_clip, print_freq=print_freq, adapt_grad_norm=adapt_grad_norm)

    if not subsplit_str:
        train_split_str = 'train'
        val_split_str = 'val'
    else:
        train_split_str = 'train_' + subsplit_str
        ploc = subsplit_str.index('l')
        val_split_str = 'train_' + subsplit_str[:ploc] + 'r' + subsplit_str[(ploc+1):]

    # Evaluation Data loading code
    eval_batch_size = eval_batch_size if eval_batch_size > 0 else batch_size
    val_data = DataRegime(getattr(model, 'data_eval_regime', None),
                          defaults={'datasets_path': datasets_dir, 'name': dataset, 'split': val_split_str, 'augment': False,
                                    'input_size': input_size, 'batch_size': eval_batch_size, 'shuffle': False,
                                    'num_workers': workers, 'pin_memory': True, 'drop_last': False})

    if evaluate:
        results = trainer.validate(val_data.get_loader())
        logging.info(results)
        return

    # Evaluation Data loading code
    if augval:
        augval_data = DataRegime(
            getattr(model, 'data_eval_regime', None),
            defaults={
                'datasets_path': datasets_dir, 'name': dataset, 'split': val_split_str, 'augment': True,
                'input_size': input_size, 'batch_size': eval_batch_size, 'shuffle': False,
                'num_workers': workers, 'pin_memory': True, 'drop_last': False,
                'autoaugment': autoaugment,
                'cutout': {'holes': 1, 'length': 16} if cutout else None,
            },
        )

    # Training Data loading code
    train_data = DataRegime(getattr(model, 'data_regime', None),
                            defaults={'datasets_path': datasets_dir, 'name': dataset, 'split': train_split_str, 'augment': True,
                                      'input_size': input_size,  'batch_size': batch_size, 'shuffle': True,
                                      'num_workers': workers, 'pin_memory': True, 'drop_last': True,
                                      'distributed': distributed, 'duplicates': duplicates, 'autoaugment': autoaugment,
                                      'cutout': {'holes': 1, 'length': 16} if cutout else None})

    logging.info('optimization regime: %s', optim_regime)
    start_epoch = max(start_epoch, 0)
    trainer.training_steps = start_epoch * len(train_data)
    for epoch in range(start_epoch, epochs):
        trainer.epoch = epoch
        train_data.set_epoch(epoch)
        val_data.set_epoch(epoch)
        if augval:
            augval_data.set_epoch(epoch)
        logging.info('\nStarting Epoch: {0}\n'.format(epoch + 1))

        # train for one epoch
        train_results = trainer.train(train_data.get_loader(),
                                      duplicates=train_data.get('duplicates'),
                                      chunk_batch=chunk_batch)

        if distributed and local_rank > 0:
            continue

        # evaluate on validation set
        val_results = trainer.validate(val_data.get_loader())
        if augval:
            augval_results = trainer.validate(augval_data.get_loader())

        # remember best prec@1 and save checkpoint
        is_best = val_results['prec1'] > best_prec1
        best_prec1 = max(val_results['prec1'], best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model,
            'config': model_config,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1
        }, is_best, path=save_path)

        logging_format = (
            '\nResults - Epoch: {0:3d}/{1:3d}\n'
            '  Training Loss   {train[loss]:.4f}   '
            'Training Prec@1   {train[prec1]:.3f}   '
            'Training Prec@5   {train[prec5]:.3f}\n'
        )
        if augval:
            logging_format += (
                '  AugVal Loss     {augval_results[loss]:.4f}   '
                'AugVal Prec@1     {augval_results[prec1]:.3f}   '
                'AugVal Prec@5     {augval_results[prec5]:.3f}\n'
            )
        logging_format += (
            '  Validation Loss {val[loss]:.4f}   '
            'Validation Prec@1 {val[prec1]:.3f}   '
            'Validation Prec@5 {val[prec5]:.3f}\n'
        )
        logging.info(
            logging_format
            .format(
                epoch + 1, epochs,
                train=train_results, val=val_results,
                augval_results=augval_results if augval else None,
            )
        )

        values = dict(epoch=epoch + 1, steps=trainer.training_steps)
        values.update({'lr': trainer.optimizer.optimizer.param_groups[0]['lr']})
        plot_partitions = ['training', 'validation']
        values.update({'training ' + k: v for k, v in train_results.items()})
        values.update({'validation ' + k: v for k, v in val_results.items()})
        if augval:
            plot_partitions.append('aug val')
            values.update({'aug val ' + k: v for k, v in augval_results.items()})
        results.add(**values)

        results.plot(x='epoch', y=[k + ' loss' for k in plot_partitions],
                     legend=plot_partitions,
                     title='Loss', ylabel='loss')
        results.plot(x='epoch', y=[k + ' error1' for k in plot_partitions],
                     legend=plot_partitions,
                     title='Error@1', ylabel='error %')
        results.plot(x='epoch', y=[k + ' error5' for k in plot_partitions],
                     legend=plot_partitions,
                     title='Error@5', ylabel='error %')
        results.plot(x='epoch', y=['lr'],
                     legend=['lr'],
                     title='Learning Rate', ylabel='Learning rate (end of epoch)')
        if 'grad' in train_results.keys():
            results.plot(x='epoch', y=['training grad'],
                         legend=['gradient L2 norm'],
                         title='Gradient Norm', ylabel='value')
        results.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')

    parser.add_argument('--results-dir', metavar='RESULTS_DIR', default='./results',
                        help='results dir')
    parser.add_argument('--save', metavar='SAVE', default='',
                        help='saved folder')
    parser.add_argument('--datasets-dir', metavar='DATASETS_DIR', default='~/Datasets',
                        help='datasets dir')
    parser.add_argument('--dataset', metavar='DATASET', default='imagenet',
                        help='dataset name or folder')
    parser.add_argument('--model', '-a', metavar='MODEL', default='alexnet',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: alexnet)')
    parser.add_argument('--input-size', type=int, default=None,
                        help='image input size')
    parser.add_argument('--model-config', default='',
                        help='additional architecture configuration')
    parser.add_argument('--dtype', default='float',
                        help='type of tensor: ' +
                        ' | '.join(torch_dtypes.keys()) +
                        ' (default: float)')
    parser.add_argument('--device', default='cuda',
                        help='device assignment ("cpu" or "cuda")')
    parser.add_argument('--device-ids', default=[0], type=int, nargs='+',
                        help='device ids assignment (e.g 0 1 2 3')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='rank of distributed processes')
    parser.add_argument('--dist-init', default='env://', type=str,
                        help='init used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=-1, type=int, metavar='N',
                        help='manual epoch number (useful on restarts). -1 for unset (will start at 0)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--eval-batch-size', default=-1, type=int,
                        help='mini-batch size (default: same as training)')
    parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                        help='optimizer function used')
    parser.add_argument('--label-smoothing', default=0, type=float,
                        help='label smoothing coefficient - default 0')
    parser.add_argument('--mixup', default=None, type=float,
                        help='mixup alpha coefficient - default None')
    parser.add_argument('--duplicates', default=1, type=int,
                        help='number of augmentations over singel example')
    parser.add_argument('--chunk-batch', default=1, type=int,
                        help='chunk batch size for multiple passes (training)')
    parser.add_argument('--cutout', action='store_true', default=False,
                        help='cutout augmentations')
    parser.add_argument('--autoaugment', action='store_true', default=False,
                        help='use autoaugment policies')
    parser.add_argument('--grad-clip', default=-1, type=float,
                        help='maximum grad norm value, -1 for none')
    parser.add_argument('--loss-scale', default=1, type=float,
                        help='loss scale for mixed precision training.')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--adapt-grad-norm', default=None, type=int,
                        help='adapt gradient scale frequency (default: None)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                        help='evaluate model FILE on validation set')
    parser.add_argument('--seed', default=123, type=int,
                        help='random seed (default: 123)')
    parser.add_argument('--subsplit-str', default='', type=str,
                        help=
                            'subsplitting string (default: none).'
                            'If given, train and val partitions are subsets of the training set.')
    parser.add_argument('--augval', action='store_true',
                        help='also show validation with augmentations')

    main(**vars(parser.parse_args()))
