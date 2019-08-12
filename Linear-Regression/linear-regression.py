# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 23:05:01 2019

@author: nitin
"""

import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

# Hyper-parameters
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001


x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)


# Convert numpy arrays to torch tensors
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

# Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)  # input and output is 1 dimension

    def forward(self, x):
        out = self.linear(x)
        return out


model = LinearRegression()

###############################################
# Linear regression model
# model = nn.Linear(input_size, output_size)
###############################################

# Define the loss function and the optimization function
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Start training
for epoch in range(num_epochs):
    inputs = Variable(x_train)
    target = Variable(y_train)    
    
    # forward
    out = model(inputs)
    loss = criterion(out, target)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 5 == 0:
        print('Epoch[{}/{}], loss: {:.6f}'
              .format(epoch+1, num_epochs, loss.data))

model.eval()
predict = model(Variable(x_train))
predict = predict.data.numpy()

# Plot diagram
plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
plt.plot(x_train.numpy(), predict, label='Fitting Line')
plt.legend() 
plt.show()

# Save the model
torch.save(model.state_dict(), './linear.pth')

