from pathlib import Path

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import datasets as dset
from torch.utils.data import random_split
from torch.utils.data import Dataset, TensorDataset

from uncertainty_eval.datasets.abstract_datasplit import DatasetSplit


class CIFAR10(DatasetSplit):
    def __init__(self, data_root, train_size=0.9, split_seed=1):
        self.data_root = data_root
        self.train_size = train_size
        self.split_seed = split_seed
        self.ds_class = dset.CIFAR10

    def train(self, transform):
        train_data = self.ds_class(self.data_root, train=True, transform=transform)
        train_size = int(len(train_data) * self.train_size)
        val_size = len(train_data) - train_size
        train_data, _ = random_split(
            train_data,
            lengths=[train_size, val_size],
            generator=torch.Generator().manual_seed(self.split_seed),
        )
        return train_data

    def val(self, transform):
        train_data = self.ds_class(self.data_root, train=True, transform=transform)
        train_size = int(len(train_data) * self.train_size)
        val_size = len(train_data) - train_size
        _, val_data = random_split(
            train_data,
            lengths=[train_size, val_size],
            generator=torch.Generator().manual_seed(self.split_seed),
        )
        return val_data

    def test(self, transform):
        test_data = self.ds_class(self.data_root, train=False, transform=transform)
        return test_data


class CIFAR100(CIFAR10):
    def __init__(self, data_root, train_size=0.9, split_seed=1):
        super().__init__(data_root, train_size, split_seed)
        self.ds_class = dset.CIFAR100


class LSUN(DatasetSplit):
    def __init__(self, data_root):
        self.data_root = data_root

    def train(self, transform):
        raise NotImplementedError

    def val(self, transform):
        raise NotImplementedError

    def test(self, transform):
        test_data = dset.LSUN(str(self.data_root / "lsun"), "test", transform=transform)
        return test_data


class SVHN(DatasetSplit):
    def __init__(self, data_root):
        self.data_root = data_root

    def train(self, transform):
        raise NotImplementedError

    def val(self, transform):
        raise NotImplementedError

    def test(self, transform):
        test_data = dset.SVHN(
            str(self.data_root / "svhn"), "test", transform=transform, download=True
        )
        return test_data


class GaussianNoise(DatasetSplit):
    def __init__(self, data_root, mean, std, length=10_000):
        self.data_root = data_root
        self.mean = mean
        self.std = std
        self.length = length

    def train(self, transform):
        return self.test(transform)

    def val(self, transform):
        return self.test(transform)

    def test(self, transform):
        return GaussianNoiseDataset(self.length, self.mean, self.std, transform)


class GaussianNoiseDataset(Dataset):
    """
    Use CIFAR-10 mean and standard deviation as default values.
    mean=(125.3, 123.0, 113.9), std=(63.0, 62.1, 66.7)
    """

    def __init__(self, length, mean, std, transform=None):
        self.transform = transform
        self.length = length
        self.dist = torch.distributions.Normal(mean, std)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = self.dist.sample()
        if len(self.mean.shape) == 3:
            img = Image.fromarray(img.numpy().astype(np.uint8))
        if self.transform is not None:
            img = self.transform(img)
        return img, -1


class UniformNoise(DatasetSplit):
    def __init__(self, data_root, low, high, length=10_000):
        self.low = low
        self.high = high
        self.length = length

    def train(self, transform):
        return self.test(transform)

    def val(self, transform):
        return self.test(transform)

    def test(self, transform):
        return UniformNoiseDataset(self.length, self.low, self.high, transform)


class UniformNoiseDataset(Dataset):
    def __init__(self, length, low, high, transform=None):
        self.low = low
        self.high = high
        self.transform = transform
        self.length = length
        self.dist = torch.distributions.Uniform(low, high)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = self.dist.sample()
        if len(self.low.shape) == 3:
            img = Image.fromarray(img.numpy().astype(np.uint8))
        if self.transform is not None:
            img = self.transform(img)
        return img, -1
