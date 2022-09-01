import torch
import numpy as np
import torch.nn as nn

from . import BasicNet


class LeNet(BasicNet):
    def __init__(self, artifact, logger, device, amp):
        super(LeNet, self).__init__(logger, device, amp)
        self.__name__ = "LeNet"
        self.artifact = artifact
        self.set_layers()

    def set_layers(self, weights=None, bias=None):
        assert weights is None and bias is None

        self.artifact.output_shape
        self.layers = {
            "Conv1": nn.Conv2d(3, 6, kernel_size=5),
            "ReLU1": nn.ReLU(),
            "MaxPool2D1": nn.MaxPool2d(2),
            "Conv2": nn.Conv2d(6, 16, kernel_size=5),
            "ReLU2": nn.ReLU(),
            "MaxPool2D2": nn.MaxPool2d(2),
            "Flatten": nn.Flatten(),
            "FC1": nn.Linear(16 * 5 * 5, 120),
            "FC2": nn.Linear(120, 84),
            "Out": nn.Linear(84, 10),
        }
        for x in self.layers:
            self.__setattr__(x, self.layers[x])
        self.to(self.device)
        
        self.nb_ReLUs = 0
        super().__setup__()

    def forward(self, x):
        self._batch_values = {}

        # workout the network
        for l in self.layers:
            x = self.layers[l](x)
            self._batch_values[l] = x
        return x

    # clear the model
    def clear(self):
        for l in list(self.layers.keys()):
            del self.layers[l]

        torch.cuda.empty_cache()
        super().__setup__()
        self.__delattr__("layers")
        self.logger.info("Model cleared.")
