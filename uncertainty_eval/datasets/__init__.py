from uncertainty_eval.datasets.image.datasets import *
from uncertainty_eval.datasets.tabular import *
from uncertainty_eval.datasets.toy import *
from uncertainty_eval.datasets.other import *
from uncertainty_eval.utils import partialclass


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
    "textures": Textures,
    "genomics": GenomicsDataset,
    "genomics-ood": OODGenomicsDataset,
    "sensorless": SensorlessDrive,
    "sensorless-ood": SensorlessDriveOOD,
    "segment": Segment,
    "segment-ood": SegmentOOD,
    "notmnist": NotMNISTSplit,
}


def get_dataset(dataset):
    try:
        ds = DATASETS[dataset]
    except KeyError as e:
        if "embedded" in dataset:
            if "genomics" in dataset:
                ds = partialclass(GenomicsEmbeddingsDataset, dataset_name=dataset)
            else:
                ds = partialclass(ImageEmbeddingDataset, dataset_name=dataset)
        else:
            raise ValueError(f"Dataset {dataset} not implemented.") from e
    return ds
