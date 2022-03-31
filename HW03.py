#!/usr/bin/env python
# coding: utf-8

# In[1]:


CUDA_LAUNCH_BLOCKING=1.


# In[2]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt  
import seaborn as sns 
from torchvision import transforms
from PIL import Image
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim


# In[3]:


from torchvision import datasets
data_path = '/hw3/'
cifar10 = datasets.CIFAR10(data_path, train = True, 
                           download = True, 
            transform =  transforms.Compose([transforms.ToTensor(), 
                            transforms.Normalize((0.4915, 0.4823, 0.4468), 
                                            (0.2470, 0.2435, 0.2616))]))
cifar10_val = datasets.CIFAR10(data_path, train = False, download = True, 
                    transform = transforms.Compose([transforms.ToTensor(), 
                            transforms.Normalize((0.4915, 0.4823, 0.4468), 
                                            (0.2470, 0.2435, 0.2616))]))


# In[4]:


################################
################################
########   Q1 - P1   ###########
################################
################################


# In[5]:


model = nn.Sequential(
            nn.Conv2d(3,16, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,8, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Linear(8 * 8 * 8, 32),
            nn.Tanh(),
            nn.Linear(32, 10))


# In[6]:


class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
    self.act1 = nn.Tanh()
    self.pool1 = nn.MaxPool2d(2)
    self.conv2 = nn.Conv2d(16,8, kernel_size=3, padding=1)
    self.act2 = nn.Tanh()
    self.pool2 = nn.MaxPool2d(2)
    self.fc1 = nn.Linear(8*8*8,32)
    self.act3 = nn.Tanh()
    self.fc2 = nn.Linear(32,10)

  def forward(self, x):
    out = self.pool1(self.act1(self.conv1(x)))
    out = self.pool2(self.act2(self.conv2(out)))
    out = out.view(-1, 8*8*8)
    out = self.act3(self.fc1(out))
    out = self.fc2(out)
    return out


# In[7]:


import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(16,8, kernel_size=3, padding=1)
    self.fc1 = nn.Linear(8*8*8, 32)
    self.fc2 = nn.Linear(32, 10)
  
  def forward(self, x):
    out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
    out = F.max_pool2d(torch.tanh(self.conv2(out)),2)
    out = out.view(-1, 8*8*8)
    out = torch.tanh(self.fc1(out))
    out = self.fc2(out)
    return out


# In[8]:


device = (torch.device('cuda:0') if torch.cuda.is_available() 
          else torch.device('cpu'))
print(f"Training on device {device}.")


# In[9]:


torch.cuda.is_available()


# In[10]:


import datetime
def training_loop(n_epochs, optimizer, model, loss_fn, 
                  train_loader):
  for epoch in range(1, n_epochs +1):
    loss_train = 0.0
    for imgs, labels in train_loader:
      outputs = model(imgs.to('cuda:0'))
      loss = loss_fn(outputs.to('cuda:0'), 
                     labels.to('cuda:0'))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      loss_train += loss.item()
    if epoch == 1 or epoch % 1 == 0:
      print('{} Epoch {}, Training Loss {}'.format(datetime.datetime.now(), 
                                    epoch, loss_train / len(train_loader)))


# In[11]:


train_loader = torch.utils.data.DataLoader(cifar10, batch_size=64, 
                                           shuffle=True)

model = Net().to('cuda:0')
optimizer = optim.SGD(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

training_loop(
    n_epochs =300,
    optimizer= optimizer,
    model = model,
    loss_fn = loss_fn,
    train_loader = train_loader,
    
)


# In[12]:


train_loader = torch.utils.data.DataLoader(cifar10, 
                    batch_size=64, shuffle=False)

val_loader = torch.utils.data.DataLoader(cifar10_val, 
                    batch_size=64, shuffle=False)

def validate(model, train_loader, val_loader):
    for name, loader in [("train", train_loader), ("val", val_loader)]:
      correct = 0
      total = 0
    
      with torch.no_grad():
        for imgs, labels in loader:
          outputs = model(imgs.to('cuda:0'))
          _, predicted = torch.max(outputs.to('cuda:0'), 
                                   dim=1)
          total += labels.shape[0]
          correct += int((predicted.to('cuda:0') == 
                          labels.to('cuda:0')).sum())
    
      print("Accuracy {}: {:.2f}".format(name, 
                                         correct/total))
  


# In[13]:


validate(model, train_loader, val_loader)


# In[ ]:


################################
################################
########   Q1 - P2   ###########
################################
################################


# In[14]:


model = nn.Sequential(
            nn.Conv2d(3,16, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,8, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Conv2d(8,4, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Linear(4*4*4, 32),
            nn.Tanh(),
            nn.Linear(32, 10))


# In[15]:


device = (torch.device('cuda:0') if torch.cuda.is_available() 
          else torch.device('cpu'))
print(f"Training on device {device}.")


# In[16]:


train_loader = torch.utils.data.DataLoader(cifar10, 
                    batch_size=64, shuffle=True)

model = Net().to('cuda:0')
optimizer = optim.SGD(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

training_loop(
    n_epochs =300,
    optimizer= optimizer,
    model = model,
    loss_fn = loss_fn,
    train_loader = train_loader,
    
)


# In[ ]:


################################
################################
########   Q2 - P1   ###########
################################
################################


# In[18]:


class NetDepth(nn.Module): 
    def __init__(self, n_chans1=32):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3,
                               padding=1)
        self.conv3 = nn.Conv2d(n_chans1 // 2, n_chans1 // 2,
                               kernel_size=3, padding=1)
        self.fc1 = nn.Linear(4 * 4 * n_chans1 // 2, 32)
        self.fc2 = nn.Linear(32, 2)
        
    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        out = F.max_pool2d(torch.relu(self.conv2(out)), 2)
        out = F.max_pool2d(torch.relu(self.conv3(out)), 2)
        out = out.view(-1, 4 * 4 * self.n_chans1 // 2)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out


# In[19]:


class NetRes(nn.Module):
    def __init__(self, n_chans1=32):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3,
                               padding=1)
        self.conv3 = nn.Conv2d(n_chans1 // 2, n_chans1 // 2,
                               kernel_size=3, padding=1)
        self.fc1 = nn.Linear(4 * 4 * n_chans1 // 2, 32)
        self.fc2 = nn.Linear(32, 2)
        
    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        out = F.max_pool2d(torch.relu(self.conv2(out)), 2)
        out1 = out
        out = F.max_pool2d(torch.relu(self.conv3(out)) + out1, 2)
        out = out.view(-1, 4 * 4 * self.n_chans1 // 2)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out


# In[20]:


class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chans, n_chans, kernel_size=3,
                              padding=1, bias=False)  # <1>
        self.batch_norm = nn.BatchNorm2d(num_features=n_chans)
        torch.nn.init.kaiming_normal_(self.conv.weight,
                                      nonlinearity='relu')  # <2>
        torch.nn.init.constant_(self.batch_norm.weight, 0.5)
        torch.nn.init.zeros_(self.batch_norm.bias)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = torch.relu(out)
        return out + x


# In[21]:


class ResNet10(nn.Module): 
    def __init__(self, n_chans1=32):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3,
                               padding=1)
        self.conv3 = nn.Conv2d(n_chans1 // 2, n_chans1 // 2,
                               kernel_size=3, padding=1)
        self.fc1 = nn.Linear(4 * 4 * n_chans1 // 2, 32)
        self.fc2 = nn.Linear(32, 2)
        
    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        out = F.max_pool2d(torch.relu(self.conv2(out)), 2)
        out1 = out
        out = F.max_pool2d(torch.relu(self.conv3(out)) + out1, 2)
        out = out.view(-1, 4 * 4 * self.n_chans1 // 2)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out


# In[22]:


model = ResNet10()
model.to('cuda:0')


# In[23]:


train_loader = torch.utils.data.DataLoader(cifar10, 
                    batch_size=64, shuffle=True)

model = ResNet10().to('cuda:0')
optimizer = optim.SGD(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

training_loop(
    n_epochs =300,
    optimizer= optimizer,
    model = model,
    loss_fn = loss_fn,
    train_loader = train_loader,
    
)


# In[29]:


train_loader = torch.utils.data.DataLoader(cifar10, 
                        batch_size=64, shuffle=False)

val_loader = torch.utils.data.DataLoader(cifar10_val, 
                        batch_size=64, shuffle=False)

validate(model, train_loader, val_loader)


# In[ ]:


################################
################################
########   Q2 - P2   ###########
################################
################################


# In[26]:


class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chans, n_chans, kernel_size=3,
                              padding=1, bias=False)  # <1>
        self.batch_norm = nn.BatchNorm2d(num_features=n_chans)
        torch.nn.init.kaiming_normal_(self.conv.weight,
                                      nonlinearity='relu')  # <2>
        torch.nn.init.constant_(self.batch_norm.weight, 0.5)
        torch.nn.init.zeros_(self.batch_norm.bias)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = torch.relu(out)
        return out + x


# In[27]:


train_loader = torch.utils.data.DataLoader(cifar10, 
                        batch_size=64, shuffle=True)

model = ResNet10().to('cuda:0')
optimizer = optim.SGD(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

training_loop(
    n_epochs =300,
    optimizer= optimizer,
    model = model,
    loss_fn = loss_fn,
    train_loader = train_loader,
    
)


# In[28]:


train_loader = torch.utils.data.DataLoader(cifar10, 
                    batch_size=64, shuffle=False)

val_loader = torch.utils.data.DataLoader(cifar10_val, 
                    batch_size=64, shuffle=False)

validate(model, train_loader, val_loader)


# In[ ]:


################################
################################
########   Q2 - p=.3   #########
################################
################################


# In[30]:


class ResNet10(nn.Module):
    def __init__(self, n_chans1=32):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3,
                               padding=1)
        self.conv3 = nn.Conv2d(n_chans1 // 2, n_chans1 // 2,
                               kernel_size=3, padding=1)
        self.fc1 = nn.Linear(4 * 4 * n_chans1 // 2, 32)
        self.fc2 = nn.Linear(32, 2)
        
    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        out = F.max_pool2d(torch.relu(self.conv2(out)), 2)
        out1 = out
        out = F.max_pool2d(torch.relu(self.conv3(out)) + out1, 2)
        out = out.view(-1, 4 * 4 * self.n_chans1 // 2)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out


# In[31]:


model = ResNet10()
model.to('cuda:0')


# In[32]:


train_loader = torch.utils.data.DataLoader(cifar10, 
                        batch_size=64, shuffle=True)

model = ResNet10()().to('cuda:0')
optimizer = optim.SGD(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

training_loop(
    n_epochs =300,
    optimizer= optimizer,
    model = model,
    loss_fn = loss_fn,
    train_loader = train_loader,
    
)


# In[33]:


train_loader = torch.utils.data.DataLoader(cifar10, 
                        batch_size=64, shuffle=False)

val_loader = torch.utils.data.DataLoader(cifar10_val, 
                        batch_size=64, shuffle=False)

validate(model, train_loader, val_loader)


# In[ ]:


################################
################################
###### Q2 - Lambda =.001   #####
################################
################################


# Problem 2 Part 2 Subsection C: Weight Decay with lambda of 0.001

# In[34]:


def training_loop(n_epochs, optimizer, model, loss_fn, 
                  train_loader):
  for epoch in range(1, n_epochs +1):
    loss_train = 0.0
    for imgs, labels in train_loader:
      outputs = model(imgs.to('cuda:0'))
      loss = loss_fn(outputs.to('cuda:0'), 
                     labels.to('cuda:0'))
      
      ambda = 0.0001
      norm = sum(p.pow(2.0).sum()
                    for p in model.parameters())
    
      loss = loss + ambda*norm

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      loss_train += loss.item()
      if epoch == 1 or epoch % 50 == 0:
        print('{} Epoch {}, Training Loss {}'.format(datetime.datetime.now(),
                                    epoch, loss_train / len(train_loader)))


# In[35]:


model = ResNet10()
model.to('cuda:0')


# In[41]:


train_loader = torch.utils.data.DataLoader(cifar10, 
                    batch_size=64, shuffle=True)

model = Net().to('cuda:0')
optimizer = optim.SGD(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

training_loop(
    n_epochs =300,
    optimizer= optimizer,
    model = model,
    loss_fn = loss_fn,
    train_loader = train_loader,
    
)


# In[42]:


train_loader = torch.utils.data.DataLoader(cifar10, 
                    batch_size=64, shuffle=False)

val_loader = torch.utils.data.DataLoader(cifar10_val, 
                    batch_size=64, shuffle=False)

validate(model, train_loader, val_loader)


# In[ ]:




