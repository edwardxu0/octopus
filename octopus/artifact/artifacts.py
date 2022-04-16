from . import Artifact


class MNIST(Artifact):
    def __init__(self, **kwargs):
        self.input_shape = [1, 28, 28]
        self.output_shape = [10]
        self.name = 'MNIST'
        self._mean = 0.1307
        self._std = 0.3081
        super().__init__(**kwargs)


class FashionMNIST(Artifact):
    def __init__(self, **kwargs):
        self.input_shape = [1, 28, 28]
        self.output_shape = [10]
        self.name = 'FashionMNIST'
        self._mean = 0
        self._std = 1
        super().__init__(**kwargs)


class CIFAR10(Artifact):
    def __init__(self, **kwargs):
        self.input_shape = [3, 32, 32]
        self.output_shape = [10]
        self.name = 'CIFAR10'
        self._mean = 0
        self._std = 1
        super().__init__(**kwargs)
