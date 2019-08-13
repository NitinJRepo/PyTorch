"""
Created on Sun Aug 11 15:15:45 2019

@author: nitin
"""
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable


# Hyper-parameters 
input_size = 784
num_classes = 10
num_epochs = 15
batch_size = 100
learning_rate = 0.001

# MNIST dataset (images and labels)
train_dataset = dsets.MNIST(root='../../data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = dsets.MNIST(root='../../data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader (input pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)



# Logistic Regression model
model = nn.Linear(input_size, num_classes)


# Check if GPU is available
use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.cuda()

# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

# Train the model
total_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        images = images.reshape(-1, 28*28)
        
        if use_gpu:
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()
        
        # Forward pass
        # When you call the model directly, the internal __call__ function is used.
        # This function manages all registered hooks and calls forward afterwards.
        # That’s also the reason you should call the model directly, because otherwise your hooks might not work etc.
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad() # You need to clear the existing gradients otherwise gradients will be accumulated to existing gradients.
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28)
        
        if use_gpu:
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
