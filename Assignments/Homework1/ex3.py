import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# model initalizatioin
myModel = nn.Linear(28*28, 10)
myLoss = nn.CrossEntropyLoss()
hyperP = dict({'numEpoch': 40, 'lr': 1e-4, 'batchSize': 4})
optimizer = torch.optim.Adam(myModel.parameters(), lr=hyperP['lr'])

# Download and load
train = DataLoader(datasets.FashionMNIST('data_fashionMNIST',
    train=True, download=True, transform=transforms.ToTensor()),
    shuffle=True, batch_size=hyperP['batchSize'])
test = DataLoader(datasets.FashionMNIST('data_fashionMNIST',
    train=False, download=True, transform=transforms.ToTensor()),
    shuffle=False, batch_size=hyperP['batchSize'])
# Illustration
label_fashion = dict([(0,'T-shirt'),(1,'trouser'),(2,'pullover'),(3,'dress'),
    (4,'coat'),(5,'sandal'),(6,'shirt'),(7,'sneaker'),(8,'bag'),(9,'boot')])

print(sum(p.numel() for p in myModel.parameters() if p.requires_grad))

# training
loss_evolution = np.zeros(hyperP['numEpoch'])
for epoch in range(hyperP['numEpoch']):
    print(' '.join(['-- epoch', str(epoch)]))
    running_loss = 0.0
    miniBatch = 0
    for X,y in train:
        optimizer.zero_grad()
        (N,_,nX,nY) = X.size()
        score = myModel(X.view(N, nX * nY))
        loss = myLoss(score, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.detach().numpy()
        miniBatch += 1
    loss_evolution[epoch] = running_loss/miniBatch
    print(' '.join(['average loss:', str(loss_evolution[epoch])]))
np.savetxt('ex3b.csv', loss_evolution, delimiter=',')
plt.plot(loss_evolution, '.')
plt.xlabel('epoch')
plt.ylabel('average loss')
plt.savefig('ex3b.pdf')
plt.close()

###### (c) draw templates ######
w = myModel.state_dict()['weight']
for item in label_fashion:
    plt.clf()
    plt.imshow(w[item].view(28, 28), vmin=-0.5, vmax=0.5, cmap='seismic')
    plt.title('template for w{:d} {:s}'.format(item, label_fashion[item]))
    plt.colorbar(extend='both')
    plt.savefig('ex3c{:d}.pdf'.format(item))
    plt.close()