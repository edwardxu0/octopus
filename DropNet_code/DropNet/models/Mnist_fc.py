import torch
import torch.nn as nn
import torch.nn.functional as F



#Network class    
class Mnist_fc(nn.Module):

    def __init__(self,plan,outputs):
        super(Mnist_fc, self).__init__()

        layers = []
        
        current_size = 784  # 28 * 28 = number of pixels in MNIST image.

        
        for size in plan:
            layers.append(nn.Linear(current_size, size))
            current_size = size


        self.fc = nn.ModuleList(layers)
        self.fcout = nn.Linear(current_size, outputs)
        

        
 
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten.
        for layer in self.fc:
            x = F.relu(layer(x))

        return self.fcout(x)
  
