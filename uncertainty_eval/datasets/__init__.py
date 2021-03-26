from uncertainty_eval.datasets.image.datasets import *
from uncertainty_eval.datasets.tabular import *
from uncertainty_eval.datasets.toy import *
from uncertainty_eval.datasets.other import *


DATASETS = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "lsun": LSUN,
    "svhn": SVHN,
    "mnist": MNIST,
    "kmnist": KMNIST,
    "fashionmnist": FashionMNIST,
    "gaussian_noise": GaussianNoise,
    "uniform_noise": UniformNoise,
    "Gaussian2D": Gaussian2D,
    "AnomalousGaussian2D": AnomalousGaussian2D,
    "TwoMoons": TwoMoons,
    "SwissRoll": SwissRoll,
    "Circles": Circles,
    "Rings": Rings,
    "PinWheel": PinWheel,
    "TwoSpirals": TwoSpirals,
    "Checkerboard": Checkerboard,
    "Line": Line,
    "Cosine": Cosine,
    "celeb-a": CelebA,
}


def get_dataset(dataset):
    try:
        ds = DATASETS[dataset]
    except KeyError as e:
        raise ValueError(f"Dataset {dataset} not implemented.") from e
    return ds
