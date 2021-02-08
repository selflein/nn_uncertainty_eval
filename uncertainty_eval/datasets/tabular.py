from pathlib import Path

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, Dataset

from uncertainty_eval.datasets.abstract_datasplit import DatasetSplit


def csv_to_dataset(csv_path: Path, feature_cols, label_col):
    csv = pd.read_csv(csv_path)
    csv = csv.dropna()

    features = torch.from_numpy(csv[feature_cols].to_numpy()).float()
    label = torch.from_numpy(csv[[label_col]].to_numpy()).long().squeeze()
    return features, label


def data_split(features, labels, splits=(0.8, 0.1, 0.1), seed=1):
    assert sum(splits) <= 1, f"Sum of splits {splits} needs to be <= 1."
    assert len(features) == len(
        labels
    ), "First dimension of features and labels has to match."
    idxs = np.random.permutation(len(features))

    lengths = np.array([int(s * len(features)) for s in splits])
    split_idxs = np.split(idxs, np.cumsum(lengths))[:-1]
    return [(features[i], labels[i]) for i in split_idxs]


class TabularDataset(Dataset):
    def __init__(self, features, labels, transforms=None):
        super().__init__()
        self.features = features
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features[idx]
        if self.transforms is not None:
            features = self.transforms(features)
        return features, self.labels[idx]


class Gaussian2D(DatasetSplit):
    def __init__(self, data_root, splits=[0.8, 0.1, 0.1], split_seed=1, **kwargs):
        super().__init__(data_root)
        features, labels = csv_to_dataset(
            Path(data_root) / "2DGaussians-0.2.csv", ["0", "1"], "2"
        )

        (
            (self.train_feats, self.train_labels),
            (self.val_feats, self.val_labels),
            (self.test_feats, self.test_labels),
        ) = data_split(features, labels, splits=splits)

    def train(self, transform):
        return TabularDataset(self.train_feats, self.train_labels, transform)

    def val(self, transform):
        return TabularDataset(self.val_feats, self.val_labels, transform)

    def test(self, transform):
        return TabularDataset(self.test_feats, self.test_labels, transform)


class AnomalousGaussian2D(DatasetSplit):
    """
    Does not have train, test, val split. Returns same dataset.
    """

    def __init__(self, data_root, **kwargs):
        super().__init__(data_root)
        self.features, self.labels = csv_to_dataset(
            Path(data_root) / "anomalous-2Ddataset.csv", ["0", "1"], "2"
        )

    def train(self, transform):
        return TabularDataset(self.features, self.labels, transform)

    def val(self, transform):
        return TabularDataset(self.features, self.labels, transform)

    def test(self, transform):
        return TabularDataset(self.features, self.labels, transform)
