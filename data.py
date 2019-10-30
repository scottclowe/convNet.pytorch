import os
import numpy as np
import torch
import torchvision.datasets as datasets
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from utils.regime import Regime
from utils.dataset import IndexedFileDataset
from preprocess import get_transform
import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

def get_dataset(name, split='train', transform=None,
                target_transform=None, download=True, datasets_path='~/Datasets'):
    is_main_pt_train = ('train' in split)
    root = os.path.join(os.path.expanduser(datasets_path), name)
    if name == 'cifar10':
        dataset = datasets.CIFAR10(
            root=root,
            train=is_main_pt_train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
    elif name == 'cifar100':
        dataset = datasets.CIFAR100(
            root=root,
            train=is_main_pt_train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
    elif name == 'mnist':
        dataset = datasets.MNIST(
            root=root,
            train=is_main_pt_train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
    elif name == 'stl10':
        dataset = datasets.STL10(
            root=root,
            split=('train' if is_main_pt_train else split),
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
    elif name in ('imagenet', 'imagenette', 'imagewoof'):
        URL_IMAGENETTE = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette.tgz'
        URL_IMAGEWOOF = 'https://s3.amazonaws.com/fast-ai-imageclas/imagewoof.tgz'
        if is_main_pt_train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'val')
        dataset = datasets.ImageFolder(
            root=root,
            transform=transform,
            target_transform=target_transform,
        )
    elif name == 'imagenet_tar':
        if is_main_pt_train:
            root = os.path.join(root, 'imagenet_train.tar')
        else:
            root = os.path.join(root, 'imagenet_validation.tar')
        dataset = IndexedFileDataset(
            root,
            extract_target_fn=(lambda fname: fname.split('/')[0]),
            transform=transform,
            target_transform=target_transform,
        )
    else:
        raise ValueError('Unrecognised dataset: {}'.format(name))
    # Check for subsampling of partition
    split_parts = split.split('_')
    if len(split_parts) <= 1:
        return dataset
    # Need to subsample the data
    seed = 0
    left_side = None
    ratio = None
    indices = np.arange(len(dataset))
    for split_part in split_parts:
        if split_part in ('train', 'val', 'test', ''):
            continue
        if split_part[0] == 's':
            # Set seed for next shuffle
            seed = int(split_part[1:])
            # Shuffle the indices
            np.random.RandomState(seed).shuffle(indices)
            continue
        if split_part[0] == 'l':
            left_side = True
        elif split_part[0] == 'r':
            left_side = False
        else:
            raise ValueError('Subsplit specification must begin with s, l, or r')
        # The rest of the string part specifies the split ratio, either
        # if left, the fraction of the partition to keep;
        # if right, (1 - fraction) to keep
        ratio = float(split_part[1:])
        # Subsample the indices, taking either the left or right partition
        n = int(round(ratio * len(dataset)))
        if left_side:
            indices = indices[:n]
        else:
            indices = indices[n:]
    # Return the appropriate subset of the data
    return torch.utils.data.Subset(dataset, indices)


_DATA_ARGS = {'name', 'split', 'transform',
              'target_transform', 'download', 'datasets_path'}
_DATALOADER_ARGS = {'batch_size', 'shuffle', 'sampler', 'batch_sampler',
                    'num_workers', 'collate_fn', 'pin_memory', 'drop_last',
                    'timeout', 'worker_init_fn'}
_TRANSFORM_ARGS = {'transform_name', 'input_size', 'scale_size', 'normalize', 'augment',
                   'cutout', 'duplicates', 'num_crops', 'autoaugment'}
_OTHER_ARGS = {'distributed'}


class DataRegime(object):
    def __init__(self, regime, defaults={}):
        self.regime = Regime(regime, defaults)
        self.epoch = 0
        self.steps = None
        self.get_loader(True)

    def get_setting(self):
        setting = self.regime.setting
        loader_setting = {k: v for k,
                          v in setting.items() if k in _DATALOADER_ARGS}
        data_setting = {k: v for k, v in setting.items() if k in _DATA_ARGS}
        transform_setting = {
            k: v for k, v in setting.items() if k in _TRANSFORM_ARGS}
        other_setting = {k: v for k, v in setting.items() if k in _OTHER_ARGS}
        transform_setting.setdefault('transform_name', data_setting['name'])
        return {'data': data_setting, 'loader': loader_setting,
                'transform': transform_setting, 'other': other_setting}

    def get(self, key, default=None):
        return self.regime.setting.get(key, default)

    def get_loader(self, force_update=False):
        if force_update or self.regime.update(self.epoch, self.steps):
            setting = self.get_setting()
            self._transform = get_transform(**setting['transform'])
            setting['data'].setdefault('transform', self._transform)
            self._data = get_dataset(**setting['data'])
            if setting['other'].get('distributed', False):
                setting['loader']['sampler'] = DistributedSampler(self._data)
                setting['loader']['shuffle'] = None
                # pin-memory currently broken for distributed
                setting['loader']['pin_memory'] = False
            self._sampler = setting['loader'].get('sampler', None)
            self._loader = torch.utils.data.DataLoader(
                self._data, **setting['loader'])
        return self._loader

    def set_epoch(self, epoch):
        self.epoch = epoch
        if self._sampler is not None and hasattr(self._sampler, 'set_epoch'):
            self._sampler.set_epoch(epoch)

    def __len__(self):
        return len(self._data)
