import torch
import numpy as np
import torch.nn as nn

from . import BasicNet


class ReLUNet(BasicNet):
    def __init__(self, artifact, layers, logger, device, amp):
        super(ReLUNet, self).__init__(logger, device, amp)
        self.__name__ = "ReLUNet"
        self.artifact = artifact
        self.layers_configs = layers
        self.dropout = False
        self.set_layers(layers)

    def set_layers(self, layers, weights=None, bias=None):
        if weights:
            assert len(weights) == len(bias)
            assert len(layers) + 1 == len(weights)

        self.nb_ReLUs = 0
        layers = (
            [np.prod(self.artifact.input_shape)]
            + layers
            + [np.prod(self.artifact.output_shape)]
        )
        self.layers = {}

        # TODO: add flatten if first layer is FC, fix this if conv layers are added
        if type(layers[1]) == int:
            layer_name = "Flatten1"
            layer = nn.Flatten()
            self.layers[layer_name] = layer
            self.__setattr__(layer_name, layer)

        for i, l in enumerate(layers[1:-1]):
            # add FC layers
            if type(l) == int:
                layer_name = f"FC{i+1}"
                layer = nn.Linear(layers[i], l)
                if weights:
                    layer.weight.data = weights[i]
                if bias:
                    layer.bias.data = bias[i]
                self.layers[layer_name] = layer
                self.__setattr__(layer_name, layer)

                # add ReLU after network
                layer_name = f"ReLU{i+1}"
                layer = nn.ReLU()
                self.nb_ReLUs += l
                self.layers[layer_name] = layer
                self.__setattr__(layer_name, layer)
            # TODO: add conv layers
            else:
                assert False

            # add a dropout layer before the output layer
            if self.dropout and i == len(layers) - 3:
                layer_name = f"Dropout1"
                layer = nn.Dropout(0.5)
                self.layers[layer_name] = layer
                self.__setattr__(layer_name, layer)

        # add output layer
        # check last hidden layer to be a FC layer
        if type(layers[-2]) == int:
            layer_name = f"Out"
            layer = nn.Linear(layers[-2], layers[-1])
            if weights:
                layer.weight.data = weights[-1]
            if bias:
                layer.bias.data = bias[-1]
            self.layers[layer_name] = layer
            self.__setattr__(layer_name, layer)
        # TODO:check last hidden layer to be a conv layer
        else:
            assert False
        super().__setup__()
        self.to(self.device)

    def forward(self, x):
        self._batch_values = {}
        # reshape input if first hidden layer is a FC layer
        # if isinstance(
        #    self.layers[list(self.layers.keys())[0]], torch.nn.modules.linear.Linear
        # ):
        # x = x.view(-1, np.prod(self.artifact.input_shape))
        # 2 x = x.flatten(1)
        # else:
        #    assert False

        # print(self.layers)

        # workout the network
        for l in self.layers:
            x = self.layers[l](x)
            self._batch_values[l] = x
        return x

    # clear the model
    def clear(self):
        for l in list(self.layers.keys()):
            del self.layers[l]
            self.__delattr__(l)

        torch.cuda.empty_cache()
        super().__setup__()
        self.__delattr__("layers")
        self.logger.info("Model cleared.")
