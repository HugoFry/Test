from testconvnet import Net
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms, datasets
import torch.optim as optim

def loaddata():
    train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    trainset = torch.utils.data.DataLoader(train, batch_size=15, shuffle=True)
    testset = torch.utils.data.DataLoader(test, batch_size=15, shuffle=True)
    return trainset, testset

trainset, testset= loaddata()


net=Net()

optimizer=optim.Adam(net.parameters(), lr=0.001)
epochs=3

#Training the network
for epoch in range(epochs):
    print(f"Starting epoch number {epoch+1}")
    for batch in tqdm(trainset):
        x,y=batch
        net.zero_grad()
        output=net(x)
        loss=F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
    print(loss)

#Test accuracy
with torch.no_grad():
    correct=0
    total=0
    for data in testset:
        x,y=data
        output=net(x)
        for i in range(len(y)):
            total+=1
            j=torch.argmax(output[i])
            if j==y[i]:
                correct+=1
    accuracy=(correct/total)*100

print(f"The network achieved an accuracy of {accuracy}%")

