import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tfrecord.torch.dataset import MultiTFRecordDataset

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


class OODGenomicsDataset(torch.utils.data.IterableDataset):
    """PyTorch Dataset implementation for the Bacteria Genomics OOD dataset (https://github.com/google-research/google-research/tree/master/genomics_ood) proposed in

    J. Ren et al., “Likelihood Ratios for Out-of-Distribution Detection,” arXiv:1906.02845 [cs, stat], Available: http://arxiv.org/abs/1906.02845.
    """

    splits = {
        "train": "before_2011_in_tr",
        "val": "between_2011-2016_in_val",
        "test": "after_2016_in_test",
        "val_ood": "between_2011-2016_ood_val",
        "test_ood": "after_2016_ood_test",
    }

    def __init__(self, data_root, split="train", transform=None, target_transform=None):
        if isinstance(data_root, str):
            self.data_root = Path(data_root)

        assert split in self.splits, f"Split '{split}' does not exist."
        split_dir = self.data_root / self.splits[split]

        tf_record_ids = [f.stem for f in split_dir.iterdir() if f.suffix == ".tfrecord"]

        self.ds = MultiTFRecordDataset(
            data_pattern=str(split_dir / "{}.tfrecord"),
            index_pattern=str(split_dir / "{}.index"),
            splits={id_: 1 / len(tf_record_ids) for id_ in tf_record_ids},
            description={"x": "byte", "y": "int", "z": "byte"},
        )

        with open(self.data_root / "label_dict.json") as f:
            label_dict = json.load(f)
            self.label_dict = {v: k for k, v in label_dict.items()}

        transform = transform if transform is not None else lambda x: x
        target_transform = (
            target_transform if target_transform is not None else lambda x: x
        )
        self.full_transform = lambda x: self.full_transform(
            x, transform, target_transform
        )

    @staticmethod
    def full_transform(item, transform, target_transform):
        x = torch.from_numpy(transform(item["x"].copy()))
        y = torch.from_numpy(target_transform(item["y"].copy()))
        return x, y

    def __iter__(self):
        return map(self.full_transform, self.ds.__iter__())
