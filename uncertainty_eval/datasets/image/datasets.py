from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import ConcatDataset
from torchvision import datasets as dset
from torch.utils.data import random_split, Dataset

from uncertainty_eval.datasets.abstract_datasplit import DatasetSplit


class CIFAR10(DatasetSplit):

    data_shape = (32, 32, 3)

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
    data_shape = (28, 28, 1)

    def __init__(self, data_root, train_size=0.9, split_seed=1):
        super().__init__(data_root, train_size, split_seed)
        self.ds_class = dset.MNIST


class FashionMNIST(CIFAR10):
    data_shape = (28, 28, 1)

    def __init__(self, data_root, train_size=0.9, split_seed=1):
        super().__init__(data_root, train_size, split_seed)
        self.ds_class = dset.FashionMNIST


class KMNIST(CIFAR10):
    data_shape = (28, 28, 1)

    def __init__(self, data_root, train_size=0.9, split_seed=1):
        super().__init__(data_root, train_size, split_seed)
        self.ds_class = dset.KMNIST


class CelebA(DatasetSplit):
    def __init__(self, data_root):
        self.data_root = data_root

    def train(self, transform):
        test_data = dset.CelebA(
            str(self.data_root), "train", transform=transform, download=True
        )
        return test_data

    def val(self, transform):
        test_data = dset.CelebA(
            str(self.data_root), "valid", transform=transform, download=True
        )
        return test_data

    def test(self, transform):
        test_data = dset.CelebA(
            str(self.data_root), "test", transform=transform, download=True
        )
        return test_data


class LSUN(DatasetSplit):
    def __init__(self, data_root):
        self.data_root = data_root

    def train(self, transform):
        test_data = dset.LSUN(
            str(self.data_root / "lsun"), "train", transform=transform
        )
        return test_data

    def val(self, transform):
        test_data = dset.LSUN(str(self.data_root / "lsun"), "val", transform=transform)
        return test_data

    def test(self, transform):
        test_data = dset.LSUN(str(self.data_root / "lsun"), "test", transform=transform)
        return test_data


class SVHN(DatasetSplit):
    def __init__(self, data_root):
        self.data_root = data_root

    def train(self, transform):
        test_data = dset.SVHN(
            str(self.data_root / "svhn"), "train", transform=transform, download=True
        )
        return test_data

    def val(self, transform):
        raise NotImplementedError

    def test(self, transform):
        test_data = dset.SVHN(
            str(self.data_root / "svhn"), "test", transform=transform, download=True
        )
        return test_data


class Textures(DatasetSplit):
    def __init__(self, data_root):
        self.data_root = data_root

    def train(self, transform):
        raise NotImplementedError

    def val(self, transform):
        raise NotImplementedError

    def test(self, transform):
        test_data = dset.ImageFolder(
            str(self.data_root / "textures" / "train"), transform=transform
        )
        valid_data = dset.ImageFolder(
            str(self.data_root / "textures" / "valid"), transform=transform
        )
        return ConcatDataset([test_data, valid_data])


class NotMNIST(dset.MNIST):

    resources = [
        (
            "https://github.com/davidflanagan/notMNIST-to-MNIST/raw/master/t10k-images-idx3-ubyte.gz",
            "2c87c839a4ef9b238846600eec8c35b7",
        ),
        (
            "https://github.com/davidflanagan/notMNIST-to-MNIST/raw/master/t10k-labels-idx1-ubyte.gz",
            "7ea9118cbafd0f6e3ee2ad771d782a01",
        ),
        (
            "https://github.com/davidflanagan/notMNIST-to-MNIST/raw/master/train-images-idx3-ubyte.gz",
            "d916d7283fce4d08db9867c640ec0042",
        ),
        (
            "https://github.com/davidflanagan/notMNIST-to-MNIST/raw/master/train-labels-idx1-ubyte.gz",
            "eab59f88903339e01dac19deed3824c0",
        ),
    ]

    classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


class NotMNISTSplit(CIFAR10):
    data_shape = (28, 28, 1)

    def __init__(self, data_root, train_size=0.9, split_seed=1):
        super().__init__(data_root, train_size, split_seed)
        self.ds_class = NotMNIST


class CIFAR10CDataset(Dataset):
    TEST_SIZE = 10_000

    def __init__(self, data_root, severity=1, corruption="all", transform=None):
        assert severity in range(1, 6)

        self.data_root = data_root / "CIFAR-10-C"
        self.data = []
        self.severity = severity
        self.transform = transform

        if corruption == "all":
            for corr_file in self.data_root.iterdir():
                if corr_file.stem != "labels":
                    self.data.append(self._extract_severity_data(corr_file))
        else:
            self.data.append(
                self._extract_severity_data(self.data_root / (corruption + ".npy"))
            )

        self.labels = self._extract_severity_data(self.data_root / "labels.npy")

    def _extract_severity_data(self, path: Path):
        arr = np.load(path)
        lower_idx = (self.severity - 1) * self.TEST_SIZE
        upper_idx = self.severity * self.TEST_SIZE
        return arr[lower_idx:upper_idx]

    def __len__(self):
        return sum(len(d) for d in self.data)

    def __getitem__(self, idx):
        data_idx = idx // self.TEST_SIZE
        subset_idx = idx % self.TEST_SIZE

        img = Image.fromarray(self.data[data_idx][subset_idx])
        if self.transform is not None:
            img = self.transform(img)

        return img, self.labels[subset_idx]


class CIFAR10C(DatasetSplit):
    data_shape = (32, 32, 3)

    def __init__(self, data_root, severity=1, corruption="all"):
        self.data_root = data_root
        self.corruption = corruption
        self.severity = severity

    def train(self, transform):
        raise NotImplementedError()

    def val(self, transform):
        raise NotImplementedError()

    def test(self, transform):
        return CIFAR10CDataset(
            self.data_root, self.severity, self.corruption, transform
        )
