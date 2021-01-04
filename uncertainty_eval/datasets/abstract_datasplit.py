import abc


class DatasetSplit(abc.ABC):
    def __init__(self, data_root):
        pass

    @abc.abstractmethod
    def train(self):
        ...

    @abc.abstractmethod
    def val(self):
        ...

    @abc.abstractmethod
    def test(self):
        ...
