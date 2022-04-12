from asyncio.log import logger
import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic_net import BasicNet


class VeriNet(BasicNet):
    def __init__(self, layers, logger, device):
        super(VeriNet, self).__init__(logger, device)
        self._set_layers(layers)
        super().__setup__()

    def _set_layers(self, layers):
        self.layers = []
        for i,l in enumerate(layers[1:-1]):
            # add FC layers
            if type(l) == int:
                layer = nn.Linear(layers[i], l)
                self.layers += [layer]
                self.__setattr__(f'FC{i+1}', layer)
                layer = nn.ReLU()
                self.layers += [layer]
                self.__setattr__(f'ReLU{i+1}', layer)
            # TODO: add conv layers
            else:
                assert False
            
            # add a dropout layer before the output layer
            if i == len(layers)-3:
                layer = nn.Dropout(0.5)
                self.Dropout1 = layer
                self.layers += [layer]
            
        # add output layer
        # check last hidden layer to be a FC layer
        if type(layers[-2]) == int:
            layer = self.OUT = nn.Linear(layers[-2], layers[-1])
            self.layers += [layer]
        # TODO:check last hidden layer to be a conv layer
        else:
            assert False
        

    def forward(self, x):
        # reshape input if first hidden layer is a FC layer
        if isinstance(self.layers[0], torch.nn.modules.linear.Linear):
            x = x.reshape(-1, 28*28)
        else:
            assert False
        
        # workout the network
        for l in self.layers:
            x = l(x)
        return x


class NetS(BasicNet):
    def __init__(self, logger, device):
        super(NetS, self).__init__(logger, device)
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc_out = nn.Linear(128, 10)
        super().__setup__()

    def forward(self, x):
        x = x.reshape(-1, 28*28)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc_out(x)
        return x


class NetM(BasicNet):
    def __init__(self, logger):
        super(NetM, self).__init__(logger)
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fcout = nn.Linear(1024, 10)
        super().__setup__()

    def forward(self, x):
        x = x.reshape(-1, 28*28)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fcout(x)
        return x


class NetL(BasicNet):
    def __init__(self, logger):
        super(NetL, self).__init__(logger)
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 1024)
        self.fc6 = nn.Linear(1024, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fcout = nn.Linear(1024, 10)
        super().__setup__()

    def forward(self, x):
        x = x.reshape(-1, 28*28)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fcout(x)
        return x


class NetConv(nn.Module):
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class Mnist_conv(nn.Module):
    def __init__(self):
        super(Mnist_conv, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4,stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4,stride=2)
        self.fc1 = nn.Linear(5*5*32, 100)
        self.fcout = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1,5*5*32)
        x = self.fc1(x)
        x = self.fcout(x)
        return x
