from . import Artifact


class MNIST(Artifact):
    def __init__(self, mode, batch_size, test_batch_size, use_cuda):
        self.input_shape = (1, 28, 28)
        self.artifact = 'MNIST'
        self._mean = 0.1307
        self._std = 0.3081
        super().__init__(mode, batch_size, test_batch_size, use_cuda)


class FashionMNIST(Artifact):
    def __init__(self, mode, batch_size, test_batch_size, use_cuda):
        self.input_shape = (1, 28, 28)
        self.artifact = 'FashionMNIST'
        self._mean = 0
        self._std = 1
        super().__init__(mode, batch_size, test_batch_size, use_cuda)


class CIFAR10(Artifact):
    def __init__(self, mode, batch_size, test_batch_size, use_cuda):
        self.input_shape = (3, 32, 32)
        self.artifact = 'CIFAR10'
        self._mean = 0
        self._std = 1
        super().__init__(mode, batch_size, test_batch_size, use_cuda)
