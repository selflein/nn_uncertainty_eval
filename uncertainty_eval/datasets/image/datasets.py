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
        train_data = self.ds_class(
            self.data_root, train=True, transform=transform, download=True
        )
        train_size = int(len(train_data) * self.train_size)
        val_size = len(train_data) - train_size
        train_data, _ = random_split(
            train_data,
            lengths=[train_size, val_size],
            generator=torch.Generator().manual_seed(self.split_seed),
        )
        return train_data

    def val(self, transform):
        train_data = self.ds_class(
            self.data_root, train=True, transform=transform, download=True
        )
        train_size = int(len(train_data) * self.train_size)
        val_size = len(train_data) - train_size
        _, val_data = random_split(
            train_data,
            lengths=[train_size, val_size],
            generator=torch.Generator().manual_seed(self.split_seed),
        )
        return val_data

    def test(self, transform):
        test_data = self.ds_class(
            self.data_root, train=False, transform=transform, download=True
        )
        return test_data


class CIFAR100(CIFAR10):
    def __init__(self, data_root, train_size=0.9, split_seed=1):
        super().__init__(data_root, train_size, split_seed)
        self.ds_class = dset.CIFAR100


class MNIST(CIFAR10):
    def __init__(self, data_root, train_size=0.9, split_seed=1):
        super().__init__(data_root, train_size, split_seed)
        self.ds_class = dset.MNIST


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
