import torch
from sklearn import datasets as sk_datasets

from uncertainty_eval.datasets.tabular import TabularDataset
from uncertainty_eval.datasets.abstract_datasplit import DatasetSplit


class TwoMoons(DatasetSplit):
    def __init__(self, data_root, length=10_000, split=(0.8, 0.1, 0.1)):
        super().__init__(data_root)
        self.train_length, self.val_length, self.test_length = [
            int(length * s) for s in split
        ]

    @staticmethod
    def _create_data(num_samples):
        x, y = sk_datasets.make_moons(n_samples=num_samples, noise=0.1)
        x = torch.from_numpy(x) * 2 + torch.tensor([-1, -0.2])
        return x.float(), torch.from_numpy(y).long()

    def train(self, transform):
        return TabularDataset(*self._create_data(self.train_length), transform)

    def val(self, transform):
        return TabularDataset(*self._create_data(self.val_length), transform)

    def test(self, transform):
        return TabularDataset(*self._create_data(self.test_length), transform)
