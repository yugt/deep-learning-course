import torch
import torch.nn as nn

###### (a) ######
class Vanilla_RNN(nn.Module):
    """
    The vanilla RNN: from (x_t,h_t-1) input,hidden-state
        h_t = tanh( R*h_t-1 + A*x_t)
        y_t = B*h_t
     where A is the encoder, B the decoder, R the recurrent matrix
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(Vanilla_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.A = nn.Linear(input_size, hidden_size, bias=False) # input to hidden
        self.R = nn.Linear(hidden_size, hidden_size, bias=False) # hidden to hidden
        self.B = nn.Linear(hidden_size, output_size, bias=False) # hidden to output

    def forward(self, x, h):
        # update the hidden state
        h_update = torch.tanh(self.R(h) + self.A(x))
        # prediction
        y = self.B(h_update)
        return y,h_update

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

myRNN = Vanilla_RNN(4, 2, 4)
x = [
torch.Tensor([1, 0, 0, 0]),
torch.Tensor([0, 1, 0, 0]),
torch.Tensor([0, 0, 1, 0]),
torch.Tensor([0, 0, 1, 0]),
torch.Tensor([0, 0, 0, 1])]
h = myRNN.init_hidden()
sd = myRNN.state_dict()
sd['A.weight'] = torch.Tensor([[1, -1, -.5, .5], [1, 1, -.5, -1]])
sd['R.weight'] = torch.Tensor([[1, 0], [0, 1]])
sd['B.weight'] = torch.Tensor([[1, 1], [.5, 1], [-1, 0], [0, -.5]])
myRNN.load_state_dict(sd)
print(myRNN.state_dict())
y = [None] * len(x)
for n in range(len(x)):
    y[n], h = myRNN(x[n], h)
chunk = [_.argmax() for _ in y]
print(y, chunk)

###### (b) ######
chunk = list(reversed(chunk))
optimizer = torch.optim.Adam(myRNN.parameters(), lr=1e-1)
criterion = nn.CrossEntropyLoss()
loss_log = []
for step in range(100):
    h = myRNN.init_hidden()
    optimizer.zero_grad()
    loss = 0.0
    chunk_predicted = []
    for p in range(len(x)):
        y, h = myRNN(x[p], h)
        target = x[len(x)-p-1].argmax().unsqueeze(0)
        loss += criterion(y, target)
        chunk_predicted.append(y.argmax())
    loss.backward()
    optimizer.step()
    if step%10 == 0:
        print(chunk_predicted)
        if(chunk_predicted==chunk): break
    loss_log.append(loss.detach().numpy() / len(chunk_predicted))
print(len(loss_log))
print(myRNN.state_dict())