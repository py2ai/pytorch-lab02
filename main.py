from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
   def __init__(self):
       super(Net, self).__init__()
       self.layer = torch.nn.Linear(1, 1)

   def forward(self, x):
       x = self.layer(x)      
       return x

net = Net()
print(net)


# Visualize our data


x = np.random.rand(100)
y = np.sin(x) * np.power(x,3) + 3*x + np.random.rand(100)*0.8

plt.scatter(x, y)
plt.show()

print('before',x.shape,y.shape)
print('after',x.reshape(-1,1).shape,y.reshape(-1,1).shape)
# convert numpy array to tensor in shape of input size
x = torch.from_numpy(x.reshape(-1,1)).float()
y = torch.from_numpy(y.reshape(-1,1)).float()
print(x.shape, y.shape)

# Define Optimizer and Loss Function
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()

inputs = Variable(x)
outputs = Variable(y)
Loss=[]

for i in range(100):

    prediction = net(inputs)
    loss = loss_func(prediction, outputs) 
    optimizer.zero_grad()
    loss.backward()        
    optimizer.step()       
    print(i)
    Loss.append(loss.data.numpy())
    if i % 10 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=2)
        plt.plot(Loss, 'b-', lw=2)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 10, 'color':  'red'})
        plt.pause(0.1)

plt.show()
torch.save(net.state_dict(),'modelsave.pth')

net = Net()
net.load_state_dict(torch.load('modelsave.pth'))
net.eval()
prediction = net(inputs)
plt.plot(prediction.data.numpy(), prediction.data.numpy(), 'b*', lw=2)
plt.show()