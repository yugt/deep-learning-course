import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

class TwoLayerNet(nn.Module):
    def __init__(self):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)
        self.fc1.weight = nn.Parameter(torch.Tensor([[3,5], [1,3]]))
        self.fc1.bias = nn.Parameter(torch.Tensor([5, 1]))
        self.fc2.weight = nn.Parameter(torch.Tensor([[1,0], [0,3]]))
        self.fc2.bias = nn.Parameter(torch.Tensor([0, 1]))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# model initialization
torch.manual_seed(0)
torch.set_deterministic(True)
device = torch.device('cpu')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hyperP = dict({'numEpoch': 40, 'lr': 1e-2, 'batchSize': 5})
myModel = TwoLayerNet()
myLoss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(myModel.parameters(), lr=hyperP['lr'])
myModel.to(device)

# load training dataset
data = np.genfromtxt('data_HW2_ex1.csv', delimiter=',')
x = data[1:, 0:-1]
y = np.int64(data[1:, -1] - 1)
x = torch.tensor(x).float().to(device)
y = torch.tensor(y).to(device)
train = DataLoader(TensorDataset(x, y), shuffle=True, batch_size=hyperP['batchSize'])
print(" The model has {:d} parameters".format(sum(p.numel()
    for p in myModel.parameters() if p.requires_grad)))

# training
loss_evolution = np.zeros(hyperP['numEpoch'])
for epoch in range(hyperP['numEpoch']):
    print(' '.join(['-- epoch', str(epoch)]))
    running_loss = 0.0
    miniBatch = 0
    for X,y in train:
        optimizer.zero_grad()
        score = myModel(X)
        loss = myLoss(score, y)
        loss.backward()
        optimizer.step()
        running_loss += torch.Tensor.cpu(loss).detach().numpy()
        miniBatch += 1
    loss_evolution[epoch] = running_loss/miniBatch
    print(' '.join(['average loss:', str(loss_evolution[epoch])]))

# test & evaluate
predict = torch.Tensor.cpu(myModel(x)).detach().numpy()
predict = (predict[:,0] < predict[:,1]).astype(np.int) 
correct = (predict == data[1:, -1] - 1)
print(np.sum(correct)/correct.size)
for name, param in myModel.named_parameters():
    if param.requires_grad:
        print(name, param.data)
color = ['red' if p==0 else 'green' for p in predict]
plt.scatter(data[1:,0], data[1:,1], c=color)
plt.savefig('ex1b.pdf')
plt.close()