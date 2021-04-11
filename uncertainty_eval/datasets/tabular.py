import abc
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


class TabularDatasetSplit(DatasetSplit):
    def __init__(self, train, val, test):
        self.train_feats, self.train_labels = train
        self.val_feats, self.val_labels = val
        self.test_feats, self.test_labels = test

    def train(self, transform):
        return TabularDataset(self.train_feats, self.train_labels, transform)

    def val(self, transform):
        return TabularDataset(self.val_feats, self.val_labels, transform)

    def test(self, transform):
        return TabularDataset(self.test_feats, self.test_labels, transform)


class Gaussian2D(TabularDatasetSplit):
    data_shape = (2,)

    def __init__(self, data_root, splits=[0.8, 0.1, 0.1], split_seed=1, **kwargs):
        features, labels = csv_to_dataset(
            Path(data_root) / "2DGaussians-0.2.csv", ["0", "1"], "2"
        )
        super().__init__(*data_split(features, labels, splits=splits))


class AnomalousGaussian2D(TabularDatasetSplit):
    data_shape = (2,)

    def __init__(self, data_root, **kwargs):
        features, labels = csv_to_dataset(
            Path(data_root) / "anomalous-2Ddataset.csv", ["0", "1"], "2"
        )
        super().__init__(*data_split(features, labels, splits=[0.0, 0.0, 1.0]))


class SensorlessDrive(TabularDatasetSplit):
    data_shape = (48,)

    def __init__(self, data_root, splits=[0.8, 0.1, 0.1], split_seed=1, **kwargs):
        features, labels = csv_to_dataset(
            Path(data_root) / "sensorless_drive_scale_10_11_missing.csv",
            [str(i) for i in range(48)],
            "48",
        )
        super().__init__(*data_split(features, labels, splits=splits))


class SensorlessDriveOOD(TabularDatasetSplit):
    data_shape = (48,)

    def __init__(self, data_root, **kwargs):
        features, labels = csv_to_dataset(
            Path(data_root) / "sensorless_drive_scale_10_11_only.csv",
            [str(i) for i in range(48)],
            "48",
        )
        super().__init__(*data_split(features, labels, splits=[0.0, 0.0, 1.0]))


class Segment(TabularDatasetSplit):
    data_shape = (18,)

    def __init__(self, data_root, splits=[0.8, 0.1, 0.1], split_seed=1, **kwargs):
        features, labels = csv_to_dataset(
            Path(data_root) / "segment_scale_sky_missing.csv",
            [str(i) for i in range(18)],
            "18",
        )
        super().__init__(*data_split(features, labels, splits=splits))


class SegmentOOD(TabularDatasetSplit):
    data_shape = (18,)

    def __init__(self, data_root, **kwargs):
        features, labels = csv_to_dataset(
            Path(data_root) / "segment_scale_sky_only.csv",
            [str(i) for i in range(18)],
            "18",
        )
        super().__init__(*data_split(features, labels, splits=[0.0, 0.0, 1.0]))
