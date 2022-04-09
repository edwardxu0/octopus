import torch
import torch.nn as nn
import torch.nn.functional as F


class Mnist_conv(nn.Module):
    def __init__(self):
        super(Mnist_conv, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32,64, kernel_size=5)
        #self.fc1 = nn.Linear(3*3*64, 256)
        self.fcout = nn.Linear(16*16*64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1,16*16*64 )
        x = self.fcout(x)
        return x