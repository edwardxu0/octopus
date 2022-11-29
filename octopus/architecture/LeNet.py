import torch
import numpy as np
import torch.nn as nn

from . import BasicNet


class LeNet(BasicNet):
    def __init__(self, artifact, net_name, logger, device, amp):
        super(LeNet, self).__init__(logger, device, amp)
        self.__name__ = "LeNet"
        self.artifact = artifact
        self.set_layers(net_name)

    def set_layers(self, net_name, weights=None, bias=None):
        assert weights is None and bias is None

        self.nb_ReLUs = 0
        self.artifact.output_shape

        # original
        if net_name == "LeNet_o":
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
        # wide(2x # kernels)
        elif net_name == "LeNet_w":
            self.layers = {
                "Conv1": nn.Conv2d(3, 12, kernel_size=5),
                "ReLU1": nn.ReLU(),
                "MaxPool2D1": nn.MaxPool2d(2),
                "Conv2": nn.Conv2d(12, 32, kernel_size=5),
                "ReLU2": nn.ReLU(),
                "MaxPool2D2": nn.MaxPool2d(2),
                "Flatten": nn.Flatten(),
                "FC1": nn.Linear(32 * 5 * 5, 120),
                "FC2": nn.Linear(120, 84),
                "Out": nn.Linear(84, 10),
            }
        # original / extended (no MaxPool2d layer)
        elif net_name == "LeNet_oe":
            self.layers = {
                "Conv1": nn.Conv2d(3, 6, kernel_size=5),
                "ReLU1": nn.ReLU(),
                # "MaxPool2D1": nn.MaxPool2d(2),
                "Conv2": nn.Conv2d(6, 16, kernel_size=5),
                "ReLU2": nn.ReLU(),
                # "MaxPool2D2": nn.MaxPool2d(2),
                "Flatten": nn.Flatten(),
                "FC1": nn.Linear(9216, 120),
                "FC2": nn.Linear(120, 84),
                "Out": nn.Linear(84, 10),
            }

        # wide (2x # kernels) / extended (no MaxPool2d layer)
        elif net_name == "LeNet_we":
            self.layers = {
                "Conv1": nn.Conv2d(3, 12, kernel_size=5),
                "ReLU1": nn.ReLU(),
                # "MaxPool2D1": nn.MaxPool2d(2),
                "Conv2": nn.Conv2d(12, 32, kernel_size=5),
                "ReLU2": nn.ReLU(),
                # "MaxPool2D2": nn.MaxPool2d(2),
                "Flatten": nn.Flatten(),
                "FC1": nn.Linear(18432, 120),
                "FC2": nn.Linear(120, 84),
                "Out": nn.Linear(84, 10),
            }
        else:
            assert False

        for i, x in enumerate(self.layers):
            l = self.layers[x]
            self.__setattr__(x, l)

            # set layers
            if i + 1 < len(self.layers.keys()):
                l_ = self.layers[list(self.layers.keys())[i + 1]]
                if isinstance(l, nn.Conv2d) and isinstance(l_, nn.ReLU):
                    print(l.out_channels)
                    self.nb_ReLUs += l.out_channels
                elif isinstance(l, nn.Linear) and isinstance(l_, nn.ReLU):
                    self.nb_ReLUs += l.out_features
                    print(l.out_features)

        self.to(self.device)
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
