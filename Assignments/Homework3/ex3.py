import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class Vanilla_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Vanilla_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.A = nn.Linear(input_size, hidden_size, bias=False) # input to hidden
        self.R = nn.Linear(hidden_size, hidden_size, bias=False) # hidden to hidden
        self.B = nn.Linear(hidden_size, output_size, bias=False) # hidden to output

    def forward(self, x, h):
        h_update = torch.tanh(self.R(h) + self.A(x))
        y = self.B(h_update)
        return y,h_update

    def perturb(self, point, dir, eps, iter=30):
        diff = []
        zero = torch.zeros(1, self.hidden_size)
        for _ in eps:
            h = zero
            y, h = self.forward(point, h)
            for __ in range(iter-1):
                y, h = self.forward(zero, h)
            h = zero
            yeps, h = self.forward(point + _ * dir, h)
            for __ in range(iter-1):
                yeps, h = self.forward(zero, h)
            diff.append(torch.linalg.norm(y-yeps))
        return torch.stack(diff, dim=0).detach().numpy()

myRNN = Vanilla_RNN(2, 2, 2)
sd = myRNN.state_dict()
sd['A.weight'] = torch.Tensor([[1, 0], [0, 1]])
sd['B.weight'] = torch.Tensor([[1, 0], [0, 1]])
sd['R.weight'] = torch.Tensor([[.5, -1], [-1, .5]])
myRNN.load_state_dict(sd)
epsilon = np.array([10**-n for n in range(4, 10)])
###### (b) ######
diff1 = myRNN.perturb(torch.Tensor([0, 0]), torch.Tensor([1, -1]), epsilon)
###### (c) ######
diff2 = myRNN.perturb(torch.Tensor([2, 1]), torch.Tensor([1, -1]), epsilon)
###### (extra) ######
diff3 = myRNN.perturb(torch.Tensor([0, 0]), torch.Tensor([1, 1]), epsilon)
print(diff1)
plt.loglog(epsilon, diff1, label='x1=(0, 0), d=(1, -1)')
plt.loglog(epsilon, diff2, label='x1=(2, 1), d=(1, -1)')
plt.loglog(epsilon, diff3, label='x1=(0, 0), d=(1, 1)')
plt.xlabel('epsilon'); plt.ylabel('difference')
plt.legend(loc='upper left')
plt.savefig("ex3.pdf")
plt.close()