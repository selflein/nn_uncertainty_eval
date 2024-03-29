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
    "Gaussian2DOverlap": Gaussian2DOverlap,
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
    "constant": Constant,
    "genomics-noise": GenomicsNoise,
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
        elif "CIFAR10-C" in dataset:
            severity = int(dataset[-1])
            ds = partialclass(CIFAR10C, severity=severity)
        else:
            raise ValueError(f"Dataset {dataset} not implemented.") from e
    return ds
