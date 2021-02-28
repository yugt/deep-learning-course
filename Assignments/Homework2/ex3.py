import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
import numpy as np

class VGG8(nn.Module):
    def __init__(self):
        super(VGG8, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features=100, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=100, out_features=10, bias=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# model initalizatioin
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hyperP = dict({'numEpoch': 10, 'lr': 1e-4, 'batchSize': 5})
if torch.cuda.is_available():
    hyperP['batchSize'] = 600
myModel = VGG8()
myLoss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(myModel.parameters(), lr=hyperP['lr'])
myModel.to(device)

# Download and load
train = DataLoader(datasets.FashionMNIST('data_fashionMNIST',
    train=True, download=True, transform=transforms.ToTensor()),
    shuffle=True, batch_size=hyperP['batchSize'])
test = DataLoader(datasets.FashionMNIST('data_fashionMNIST',
    train=False, download=True, transform=transforms.ToTensor()),
    shuffle=False, batch_size=hyperP['batchSize'])

print(sum(p.numel() for p in myModel.parameters() if p.requires_grad))

def testing(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in testloader:
            images = X.to(device)
            labels = y.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy: {:4f}%'.format(100 * correct / total))

try:
    myModel = torch.load("ex3.pt")
    myModel.eval()
    testing(myModel, test)
except:
    # training
    loss_evolution = np.zeros(hyperP['numEpoch'])
    for epoch in range(hyperP['numEpoch']):
        print(' '.join(['-- epoch', str(epoch)]))
        running_loss = 0.0
        miniBatch = 0
        for X,y in train:
            optimizer.zero_grad()
            score = myModel(X.to(device))
            loss = myLoss(score, y.to(device))
            loss.backward()
            optimizer.step()
            running_loss += torch.Tensor.cpu(loss).detach().numpy()
            miniBatch += 1
        loss_evolution[epoch] = running_loss/miniBatch
        print("average loss: {:4f}".format(loss_evolution[epoch]))
    torch.save(myModel, "ex3.pt")
