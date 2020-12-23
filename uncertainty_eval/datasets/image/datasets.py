import abc
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import datasets as dset
from torch.utils.data import random_split
from torch.utils.data import Dataset, TensorDataset


class DatasetSplit(abc.ABC):
    def __init__(self, data_root):
        pass

    @abc.abstractmethod
    def train(self):
        ...

    @abc.abstractmethod
    def val(self):
        ...

    @abc.abstractmethod
    def test(self):
        ...


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


class TabularDataset(DatasetSplit):
    def __init__(
        self,
        csv_path,
        splits=[0.8, 0.1, 0.1],
        split_seed=1,
        label_col="2",
        feature_cols=["0", "1"],
    ):
        csv = pd.read_csv(csv_path)
        csv = csv.dropna()

        features = torch.from_numpy(csv[feature_cols].to_numpy()).float()
        label = torch.from_numpy(csv[[label_col]].to_numpy()).long().squeeze()
        ds = TensorDataset(features, label)

        self.train_data, self.val_data, self.test_data = random_split(
            ds,
            lengths=[int(len(ds) * split) for split in splits],
            generator=torch.Generator().manual_seed(split_seed),
        )

    def train(self, transform):
        return self.train_data

    def val(self, transform):
        return self.val_data

    def test(self, transform):
        return self.test_data


class Gaussian2D(TabularDataset):
    def __init__(self, data_root, **kwargs):
        super().__init__(
            Path(data_root) / "2DGaussians-0.2.csv",
            label_col="2",
            feature_cols=["0", "1"],
            **kwargs,
        )


class AnomalousGaussian2D(TabularDataset):
    def __init__(self, data_root, **kwargs):
        super().__init__(
            Path(data_root) / "anomalous-2Ddataset.csv",
            label_col="2",
            feature_cols=["0", "1"],
            splits=[0.0, 0.0, 1.0] ** kwargs,
        )


DATASETS = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "lsun": LSUN,
    "svhn": SVHN,
    "gaussian_noise": GaussianNoise,
    "uniform_noise": UniformNoise,
    "Gaussian2D": Gaussian2D,
    "AnomalousGaussian2D": AnomalousGaussian2D,
}


def get_dataset(dataset):
    try:
        ds = DATASETS[dataset]
    except KeyError as e:
        raise ValueError(f"Dataset {dataset} not implemented.") from e
    return ds
