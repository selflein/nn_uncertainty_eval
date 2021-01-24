import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from uncertainty_eval.datasets.abstract_datasplit import DatasetSplit


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
        self.mean = mean
        self.std = std
        self.length = length
        self.dist = torch.distributions.Normal(mean, std)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = self.dist.sample()
        if len(self.mean.shape) == 3:
            img = Image.fromarray(img.numpy().squeeze().astype(np.uint8))
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
            img = Image.fromarray(img.numpy().squeeze().astype(np.uint8))
        if self.transform is not None:
            img = self.transform(img)
        return img, -1
