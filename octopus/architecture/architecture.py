import torch
import numpy as np
import torch.nn as nn

from .basic_net import BasicNet


class VeriNet(BasicNet):
    def __init__(self, artifact, layers, logger, device, amp):
        super(VeriNet, self).__init__(logger, device, amp)
        self.artifact = artifact
        self.set_layers(layers)
        super().__setup__()


    def set_layers(self, layers, weights=None, bias=None):
        self.layers = {}
        for i,l in enumerate(layers[1:-1]):
            # add FC layers
            if type(l) == int:
                layer_name = f'FC{i+1}'
                layer = nn.Linear(layers[i], l)
                if weights:
                    layer.weight.data = weights[i]
                if bias:
                    layer.weight.data = bias[i]
                self.layers[layer_name] = layer
                self.__setattr__(layer_name, layer)

                layer_name = f'ReLU{i+1}'
                layer = nn.ReLU()
                self.layers[layer_name] = layer
                self.__setattr__(layer_name, layer)
            # TODO: add conv layers
            else:
                assert False
            
            # add a dropout layer before the output layer
            if i == len(layers)-3:
                layer_name = f'Dropout1'
                layer = nn.Dropout(0.5)
                self.layers[layer_name] = layer
                self.__setattr__(layer_name, layer)
            
        # add output layer
        # check last hidden layer to be a FC layer
        if type(layers[-2]) == int:
            layer_name = f'Out'
            layer = nn.Linear(layers[-2], layers[-1])
            if weights:
                layer.weight.data = weights[-1]
            if bias:
                layer.weight.data = bias[-1]
            self.layers[layer_name] = layer
            self.__setattr__(layer_name, layer)
        # TODO:check last hidden layer to be a conv layer
        else:
            assert False
        

    def forward(self, x):
        # reshape input if first hidden layer is a FC layer
        if isinstance(self.layers[list(self.layers.keys())[0]], torch.nn.modules.linear.Linear):
            x = x.view(-1, np.prod(self.artifact.input_shape))
        else:
            assert False
        
        # workout the network
        for l in self.layers:
            x = self.layers[l](x)
        return x

    # clear the model
    def clear(self):
        for l in list(self.layers.keys()):
            del(self.layers[l])
            self.__delattr__(l)
    
        torch.cuda.empty_cache()
        self.__delattr__('layers')
        self.logger.info('Model cleared.')
