import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #I don't understand how this architecture works
        self.conv1 = nn.Conv2d(1,32,5)
        self.conv2 = nn.Conv2d(32, 64,5)

        #checks the output size of the convolutional layers
        randomimage=torch.rand(28,28).view(1,28,28)
        output=self.conv(randomimage)
        self.outputsize=output.shape[0]*output.shape[1]*output.shape[2]

        self.fc1 = nn.Linear(self.outputsize,256)
        self.fc2 = nn.Linear(256,10)

    def conv(self,x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,(2,2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))
        return x

    def forward(self,x):
        x=self.conv(x)
        x=x.view([-1,self.outputsize])
        x=F.relu(self.fc1(x))
        x=F.log_softmax(self.fc2(x),dim=1)
        return x
