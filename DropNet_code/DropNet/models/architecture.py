import torch
import torch.nn as nn
import torch.nn.functional as F

class NetS(nn.Module):
    def __init__(self):
        super(NetS, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fcout = nn.Linear(128, 10)

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

class NetM(nn.Module):
    def __init__(self):
        super(NetM, self).__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fcout = nn.Linear(1024, 10)

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

class NetL(nn.Module):
    def __init__(self):
        super(NetL, self).__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 1024)
        self.fc6 = nn.Linear(1024, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fcout = nn.Linear(1024, 10)

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