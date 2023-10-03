import torch
import torch.nn as nn
import numpy as np

from . import BasicNet


class CIFAR2020(BasicNet):
    def __init__(self, artifact, net_name, logger, device, amp):
        super(CIFAR2020, self).__init__(logger, device, amp)
        self.__name__ = "CIFAR2020"
        self.artifact = artifact
        self.net_name = net_name
        self.set_layers(net_name)

    def set_layers(self, net_name, weights=None, bias=None):
        assert weights is None and bias is None

        self.nb_ReLUs = 0
        self.artifact.output_shape

        # original
        if net_name == "CIFAR2020_2_255":
            self.layers = {
                "Conv1": nn.Conv2d(3, 32, 3, stride=1, padding=1),
                "ReLU1": nn.ReLU(),
                "Conv2": nn.Conv2d(32, 32, 4, stride=2, padding=1),
                "ReLU2": nn.ReLU(),
                "Conv3": nn.Conv2d(32, 128, 4, stride=2, padding=1),
                "ReLU3": nn.ReLU(),
                "Flatten": nn.Flatten(),
                "FC1": nn.Linear(128 * 8 * 8, 250),
                "ReLU4": nn.ReLU(),
                "Out": nn.Linear(250, 10),
            }
        elif net_name == "CIFAR2020_8_255":
            self.layers = {
                "Conv1": nn.Conv2d(3, 32, 5, stride=2, padding=2),
                "ReLU1": nn.ReLU(),
                "Conv2": nn.Conv2d(32, 128, 4, stride=2, padding=1),
                "ReLU2": nn.ReLU(),
                "Flatten": nn.Flatten(),
                "FC1": nn.Linear(128 * 8 * 8, 250),
                "ReLU3": nn.ReLU(),
                "Out": nn.Linear(250, 10),
            }
        elif net_name == "convBig":
            self.layers = {
                "Conv1": nn.Conv2d(3, 32, 3, stride=1, padding=1),
                "ReLU1": nn.ReLU(),
                "Conv2": nn.Conv2d(32, 32, 4, stride=2, padding=1),
                "ReLU2": nn.ReLU(),
                "Conv3": nn.Conv2d(32, 64, 3, stride=1, padding=1),
                "ReLU3": nn.ReLU(),
                "Conv4": nn.Conv2d(64, 64, 4, stride=2, padding=1),
                "ReLU4": nn.ReLU(),
                "Flatten": nn.Flatten(),
                "FC1": nn.Linear(64 * 8 * 8, 512),
                "ReLU5": nn.ReLU(),
                "FC2": nn.Linear(512, 512),
                "ReLU6": nn.ReLU(),
                "Out": nn.Linear(512, 10),
            }
        else:
            assert False

        self.eval()
        data = torch.randn([1, *self.artifact.input_shape])

        for i, x in enumerate(self.layers):
            l = self.layers[x]
            self.__setattr__(x, l)
            data = l(data)
            # set layers
            if i + 1 < len(self.layers.keys()):
                l_ = self.layers[list(self.layers.keys())[i + 1]]
                if isinstance(l, nn.Conv2d) and isinstance(l_, nn.ReLU):
                    self.nb_ReLUs += np.prod(data.shape)
                elif isinstance(l, nn.Linear) and isinstance(l_, nn.ReLU):
                    self.nb_ReLUs += l.out_features

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
