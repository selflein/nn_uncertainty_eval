import torch
import numpy as np
from PIL import Image
from torchvision import datasets as dset
from torch.utils.data import random_split
from torch.utils.data import Dataset


class CIFAR10:
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


class LSUN:
    def __init__(self, data_root):
        self.data_root = data_root

    def test(self, transform):
        test_data = dset.LSUN(str(self.data_root / "lsun"), "test", transform=transform)
        return test_data


class SVHN:
    def __init__(self, data_root):
        self.data_root = data_root

    def test(self, transform):
        test_data = dset.SVHN(
            str(self.data_root / "svhn"), "test", transform=transform, download=True
        )
        return test_data


class GaussianNoise:
    def __init__(self, data_root=None, length=10_000, shape=(32, 32), mean=(125.3, 123.0, 113.9), std=(63.0, 62.1, 66.7)):
        self.data_root = data_root
        self.shape = shape
        self.mean = mean
        self.std = std
        self.length = length

    def test(self, transform):
        return GaussianNoiseDataset(self.shape, self.length, self.mean, self.std, transform)


class GaussianNoiseDataset(Dataset):
    """
    Use CIFAR-10 mean and standard deviation as default values.
    """
    def __init__(self, shape, length, mean=(125.3, 123.0, 113.9), std=(63.0, 62.1, 66.7), transform=None):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.shape = torch.Size(shape)
        self.transform = transform
        self.length = length

        self.dist = torch.distributions.Normal(
            self.mean.unsqueeze(0).repeat(self.shape.numel(), 1).reshape(*self.shape, len(mean)),
            self.std.unsqueeze(0).repeat(self.shape.numel(), 1).reshape(*self.shape, len(std)),
        )
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = self.dist.sample()
        img = Image.fromarray(img.numpy().astype(np.uint8))
        if self.transform is not None:
            img = self.transform(img)
        return img, -1



class UniformNoise:
    def __init__(self, data_root=None, length=10_000, shape=(32, 32, 3), low=0., high=255.):
        self.data_root = data_root
        self.shape = shape
        self.low = low
        self.high = high
        self.length = length

    def test(self, transform):
        return UniformNoiseDataset(self.shape, self.length, self.low, self.high, transform)


class UniformNoiseDataset(Dataset):
    def __init__(self, shape, length, low=0, high=255, transform=None):
        self.low = low
        self.high = high
        self.shape = shape
        self.transform = transform
        self.length = length
        self.dist = torch.distributions.Uniform(
            torch.empty(*shape).fill_(low),
            torch.empty(*shape).fill_(high)
        )
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = self.dist.sample()
        img = Image.fromarray(img.numpy().astype(np.uint8))
        if self.transform is not None:
            img = self.transform(img)
        return img, -1


DATASETS = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "lsun": LSUN,
    "svhn": SVHN,
    "gaussian_noise": GaussianNoise,
    "uniform_noise": UniformNoise
}


def get_dataset(dataset):
    try:
        ds = DATASETS[dataset]
    except KeyError as e:
        raise ValueError(f"Dataset {dataset} not implemented.") from e
    return ds
