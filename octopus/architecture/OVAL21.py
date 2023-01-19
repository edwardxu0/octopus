import torch
import torch.nn as nn

from . import BasicNet


class OVAL21(BasicNet):
    def __init__(self, artifact, net_name, logger, device, amp):
        super(OVAL21, self).__init__(logger, device, amp)
        self.__name__ = "OVAL21"
        self.artifact = artifact
        self.set_layers(net_name)

    def set_layers(self, net_name, weights=None, bias=None):
        assert weights is None and bias is None

        self.nb_ReLUs = 0
        self.artifact.output_shape

        # original
        if net_name == "OVAL21_o":
            self.layers = {
                "Conv1": nn.Conv2d(3, 8, 4, stride=2, padding=1),
                "ReLU1": nn.ReLU(),
                "Conv2": nn.Conv2d(8, 16, 4, stride=2, padding=1),
                "ReLU2": nn.ReLU(),
                "Flatten": nn.Flatten(),
                "FC1": nn.Linear(16 * 8 * 8, 100),
                "ReLU3": nn.ReLU(),
                "Out": nn.Linear(100, 10),
            }
        # wide(2x # kernels)
        elif net_name == "OVAL21_w":
            self.layers = {
                "Conv1": nn.Conv2d(3, 16, 4, stride=2, padding=1),
                "ReLU1": nn.ReLU(),
                "Conv2": nn.Conv2d(16, 32, 4, stride=2, padding=1),
                "ReLU2": nn.ReLU(),
                "Flatten": nn.Flatten(),
                "FC1": nn.Linear(32 * 8 * 8, 100),
                "ReLU3": nn.ReLU(),
                "Out": nn.Linear(100, 10),
            }
        # original / extended (no MaxPool2d layer)
        elif net_name == "OVAL21_d":
            self.layers = {
                "Conv1": nn.Conv2d(3, 8, 4, stride=2, padding=1),
                "ReLU1": nn.ReLU(),
                "Conv2": nn.Conv2d(8, 8, 3, stride=1, padding=1),
                "ReLU2": nn.ReLU(),
                "Conv3": nn.Conv2d(8, 8, 3, stride=1, padding=1),
                "ReLU3": nn.ReLU(),
                "Conv4": nn.Conv2d(8, 8, 4, stride=2, padding=1),
                "ReLU4": nn.ReLU(),
                "Flatten": nn.Flatten(),
                "FC1": nn.Linear(8 * 8 * 8, 100),
                "ReLU5": nn.ReLU(),
                "Out": nn.Linear(100, 10),
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
