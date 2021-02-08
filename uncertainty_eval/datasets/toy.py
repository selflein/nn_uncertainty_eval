"""Adapted from https://github.com/wgrathwohl/VERA/blob/main/utils/toy_data.py"""

import abc

import torch
import sklearn
import numpy as np
from sklearn import datasets as sk_datasets
from sklearn.utils import shuffle as util_shuffle

from uncertainty_eval.datasets.tabular import TabularDataset
from uncertainty_eval.datasets.abstract_datasplit import DatasetSplit


class ToyDatasetSplit(DatasetSplit):
    def __init__(self, data_root, length=10_000, split=(0.8, 0.1, 0.1)):
        super().__init__(data_root)
        self.train_length, self.val_length, self.test_length = [
            int(length * s) for s in split
        ]

    @abc.abstractstaticmethod
    def _create_data(num_samples):
        pass

    def train(self, transform):
        return TabularDataset(*self._create_data(self.train_length), transform)

    def val(self, transform):
        return TabularDataset(*self._create_data(self.val_length), transform)

    def test(self, transform):
        return TabularDataset(*self._create_data(self.test_length), transform)


class TwoMoons(ToyDatasetSplit):
    @staticmethod
    def _create_data(num_samples):
        x, y = sk_datasets.make_moons(n_samples=num_samples, noise=0.1)
        x = torch.from_numpy(x) * 2 + torch.tensor([-1, -0.2])
        return x.float(), torch.from_numpy(y).long()


class SwissRoll(ToyDatasetSplit):
    @staticmethod
    def _create_data(num_samples):
        data = sk_datasets.make_swiss_roll(n_samples=num_samples, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return torch.from_numpy(data), torch.empty(data.shape[0]).fill_(-1).long()


class Circles(ToyDatasetSplit):
    @staticmethod
    def _create_data(num_samples):
        data = sk_datasets.make_circles(n_samples=num_samples, factor=0.5, noise=0.08)[
            0
        ]
        data = data.astype("float32")
        data *= 3
        return torch.from_numpy(data), torch.empty(data.shape[0]).fill_(-1).long()


class Rings(ToyDatasetSplit):
    @staticmethod
    def _create_data(num_samples):
        obs = num_samples
        num_samples *= 20
        n_samples4 = n_samples3 = n_samples2 = num_samples // 4
        n_samples1 = num_samples - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        X = (
            np.vstack(
                [
                    np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
                    np.hstack([circ4_y, circ3_y, circ2_y, circ1_y]),
                ]
            ).T
            * 3.0
        )
        X = util_shuffle(X)

        # Add noise
        X += np.random.normal(scale=0.08, size=X.shape)
        inds = np.random.choice(list(range(num_samples)), obs)
        X = X[inds]
        return torch.from_numpy(X), torch.empty(X.shape[0]).fill_(-1).long()


class PinWheel(ToyDatasetSplit):
    @staticmethod
    def _create_data(num_samples):
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = num_samples // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = np.random.randn(num_classes * num_per_class, 2) * np.array(
            [radial_std, tangential_std]
        )
        features[:, 0] += 1.0
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack(
            [np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)]
        )
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        data = 2 * np.random.permutation(np.einsum("ti,tij->tj", features, rotations))
        return torch.from_numpy(data), torch.empty(data.shape[0]).fill_(-1).long()


class TwoSpirals(ToyDatasetSplit):
    @staticmethod
    def _create_data(num_samples):
        n = np.sqrt(np.random.rand(num_samples // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(num_samples // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(num_samples // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        return torch.from_numpy(x), torch.empty(x.shape[0]).fill_(-1).long()


class Checkerboard(ToyDatasetSplit):
    @staticmethod
    def _create_data(num_samples):
        x1 = np.random.rand(num_samples) * 4 - 2
        x2_ = np.random.rand(num_samples) - np.random.randint(0, 2, num_samples) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        data = np.concatenate([x1[:, None], x2[:, None]], 1) * 2
        return (
            torch.from_numpy(data).float(),
            torch.empty(data.shape[0]).fill_(0).long(),
        )


class Line(ToyDatasetSplit):
    @staticmethod
    def _create_data(num_samples):
        x = np.random.rand(num_samples) * 5 - 2.5
        y = x
        data = np.stack((x, y), 1)
        return torch.from_numpy(data), torch.empty(data.shape[0]).fill_(-1).long()


class Cosine(ToyDatasetSplit):
    @staticmethod
    def _create_data(num_samples):
        x = np.random.rand(num_samples) * 5 - 2.5
        y = np.sin(x) * 2.5
        data = np.stack((x, y), 1)
        return torch.from_numpy(data), torch.empty(data.shape[0]).fill_(-1).long()
