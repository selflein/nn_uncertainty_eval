from uncertainty_eval.datasets.image.datasets import *
from uncertainty_eval.datasets.tabular import *


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
