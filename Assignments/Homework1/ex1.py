import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data_HW1_ex1.csv', delimiter=',')

###### (a) ######
mse = np.zeros(13)

for k in range(mse.size):
    p = np.polyfit(data[:,0], data[:,1], deg=k, full=True)
    mse[k] = p[1] / data.shape[0]

plt.plot(np.arange(mse.size), mse, '.')
plt.xlabel('k'); plt.ylabel('mse')
plt.title('loss l(P_k)')
plt.savefig("ex1a.pdf")
plt.close()

###### (b) ######
mse = np.zeros((13, 2))
train = 4 * data.shape[0] // 5 # train = 80

for k in range(mse.shape[0]):
    p = np.polyfit(data[:train, 0], data[:train, 1], deg=k, full=True)
    mse[k, 0] = p[1] / train
    p = np.poly1d(p[0])
    mse[k, 1] = (np.square(p(data[train:, 0])-data[train:, 1])).mean()

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('k')
ax1.set_ylabel('mse_train', color=color)
ax1.semilogy(np.arange(mse.shape[0]), mse[:, 0], '.', color=color)
ax1.tick_params(axis='y', labelcolor=color)

color = 'tab:red'
ax2 = ax1.twinx()
ax2.set_ylabel('mse_test', color=color)
ax2.semilogy(np.arange(mse.shape[0]), mse[:, 1], '.', color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('l_train vs l_test')
plt.savefig('ex1b.pdf')
plt.close()

###### (c) ######
# k* = 2
p = np.polyfit(data[:, 0], data[:, 1], deg=2, full=True)
print(p[0]) # [ 0.35333775 -0.27947762  0.81043897]
p = np.poly1d(p[0])
x = np.linspace(data[:,0].min(), data[:, 0].max())
plt.plot(x, p(x), '-.', color='orange')
plt.scatter(data[:,0], data[:, 1], s=10)
plt.xlabel('x'); plt.ylabel('y')
plt.savefig('ex1c.pdf')