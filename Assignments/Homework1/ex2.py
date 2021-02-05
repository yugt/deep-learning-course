from typing import no_type_check_decorator
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data_HW1_ex1.csv', delimiter=',')

# [ x_bar y_bar x^2_bar y^2_bar xy_bar ]
bar = np.array([data[:, 0].mean(), data[:, 1].mean(),
np.square(data[:, 0]).mean(), np.square(data[:, 1]).mean(),
np.multiply(data[:, 0], data[:, 1]).mean()])
# [ a*, b* ]
star = np.array([bar[2]*bar[1]-bar[0]*bar[4], bar[4]-bar[0]*bar[1]])/\
        (bar[2] - bar[0]**2)

def loss(theta):
    a = theta[0]; b = theta[1]
    return a**2 + np.multiply(bar,
    np.array([2*a*b, -2*a, b**2, 1, -2*b])).sum()

def grad_loss(theta):
    a = theta[0]; b = theta[1]
    return 2 * np.array([a + b*bar[0] - bar[1],
    b*bar[2] + a*bar[0] - bar[4]])

def estimate_conv_rate(convergence, figname):
    rate = np.concatenate((np.zeros(1), np.diff(np.log(convergence))))
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('||(a_n, b_n) - (a*, b*)||', color=color)
    ax1.semilogy(convergence, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    color = 'tab:red'
    ax2 = ax1.twinx()
    ax2.set_ylabel('||(a_n, b_n) - (a*, b*)||/\
    ||(a_n-1, b_n-1) - (a*, b*)||''', color=color)
    ax2.semilogy(np.exp(rate), '.', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Gradient descent convergence')
    plt.savefig(figname)
    plt.close()

###### (b) ######
learning_rate = 0.05; epoch = 1000
theta_saved = np.zeros((epoch+1, 2))
theta_saved[0,] = np.array([1.8, 1.4])
loss_saved = np.zeros(epoch+1)
loss_saved[0] = loss(theta_saved[0,])

for k in range(epoch):
    g = grad_loss(theta_saved[k])
    theta_saved[k+1,] = theta_saved[k,] - learning_rate * g
    loss_saved[k+1] = loss(theta_saved[k+1,])

estimate_conv_rate(np.sqrt(np.square(theta_saved - star).sum(axis=1)), 'ex2b.pdf')

###### (c) ######
def momentum(theta, v, lr=learning_rate, gamma=0.9):
    v = gamma * v + lr * grad_loss(theta)
    theta = theta - v
    return (theta, v)

def nesterov(theta, v, lr=learning_rate, gamma=0.9):
    v = gamma * v + lr * grad_loss(theta - gamma * v)
    theta = theta - v
    return (theta, v)

v = theta_saved[0,]
for k in range(epoch):
    (t, v) = momentum(theta_saved[k], v)
    theta_saved[k+1,] = t
    loss_saved[k+1] = loss(t)
estimate_conv_rate(np.sqrt(np.square(theta_saved - star).sum(axis=1)), 'ex2cm.pdf')

v = theta_saved[0,]
for k in range(epoch):
    (t, v) = nesterov(theta_saved[k], v)
    theta_saved[k+1,] = t
    loss_saved[k+1] = loss(t)
estimate_conv_rate(np.sqrt(np.square(theta_saved - star).sum(axis=1)), 'ex2cn.pdf')