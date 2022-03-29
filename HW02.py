#!/usr/bin/env python
# coding: utf-8

# In[1]:

#Homework 2
#Name : Abrar Altaay
#ID: 801166376
# https://github.com/aaltaay1/Intro-to-Machine-Learning.git

from torchvision import transforms
from PIL import Image
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt  
import seaborn as sns 
from ptflops import get_model_complexity_info


# In[2]:


#Reading training data
housing = pd.DataFrame(pd.read_csv("Housing.csv")) 
housing.head() 


# In[3]:


num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price'] 
Newtrain = housing[num_vars] 


# In[4]:


# #Price Predicition
# t_area = torch.tensor(Newtrain['area'])
# t_bedrooms = torch.tensor(Newtrain['bedrooms'])
# t_bathrooms = torch.tensor(Newtrain['bathrooms'])
# t_stories = torch.tensor(Newtrain['stories'])
# t_parking = torch.tensor(Newtrain['parking'])
# t_price = torch.tensor(Newtrain['price'])
# #Normalized
# t_areaN = t_area * 0.1
# t_bedroomsN = t_bedrooms * 0.1
# t_bathroomsN = t_bathrooms * 0.1
# t_storiesN = t_stories * 0.1
# t_parkingN = t_parking * 0.1
# def model(t_area, t_bedrooms, t_bathrooms, t_stories, t_parking, w1, w2, w3, w4, w5, b):
#     return w5*t_parking + w4*t_stories + w3*t_bathrooms + w2*t_bedrooms + w1*t_area + b


# In[5]:


from sklearn.model_selection import train_test_split 
 
# We specify this so that the train and test data set always have the same rows, respec
np.random.seed(0) 
df_train, df_test = train_test_split(Newtrain, train_size = 0.8, test_size = 0.2, random_state=33)


# In[6]:


num_vas = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

Train_Output = df_train.pop('price')
Test_Out = df_test.pop('price')

Train_Arr = df_train.to_numpy()
newTrainOut = Train_Output.to_numpy()
testArray = df_test.to_numpy()
newTestOut = Test_Out.to_numpy()


# In[7]:


import warnings 
warnings.filterwarnings('ignore') 
 
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
 

scaler = StandardScaler() 
scaler = MinMaxScaler() 
Newtrain[num_vars] = scaler.fit_transform(Newtrain[num_vars]) 


# In[8]:


trans = torch.tensor


# In[9]:


Train_Data = trans(Train_Arr)
Train_Outputput = trans(newTrainOut)
Test_Data = trans(testArray)
Test_Output = trans(newTestOut)


# In[10]:


print(Train_Data)


# In[11]:


Train_Data = Train_Data.to(torch.float32)
Train_Outputput = Train_Outputput.to(torch.float32)
Test_Data = Test_Data.to(torch.float32)
Test_Output = Test_Output.to(torch.float32)


# In[12]:


T_Mean = torch.mean(Train_Data)
outputMean = torch.mean(Train_Outputput)


# In[13]:


trainSTD = torch.std(Train_Data)
outputSTD = torch.std(Train_Outputput)


# In[14]:


Train_Data = (Train_Data-T_Mean)/trainSTD
Train_Outputput = (Train_Outputput-outputMean)/outputSTD


# In[15]:


Train_Outputput = torch.tensor(Train_Outputput).unsqueeze(1)
Test_Output = torch.tensor(Test_Output).unsqueeze(1)


# In[16]:


seq_model = nn.Sequential(
            nn.Linear(5,8),
            nn.Tanh(),
            nn.Linear(8,1))


# In[17]:


optimizer = optim.SGD(
        seq_model.parameters(),
        lr=1e-1)


# In[18]:


def training_loop(n_epochs, optimizer, model, loss_fn, t_u_train, t_u_val, t_c_train, t_c_val):
    for epoch in range(1, n_epochs + 1):
        t_p_train = model(t_u_train)
        loss_train = loss_fn(t_p_train, t_c_train)
        
        t_p_val = model(t_u_val)
        
        loss_val = loss_fn(t_p_val, t_c_val)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        if epoch == 1 or epoch % 100 == 0:
            print(f"Epoch {epoch}, Training loss {loss_train.item():0.4f},"f" Validation loss {loss_val.item():.4f}")
            


# In[19]:


training_loop(
    n_epochs = 3000,
    optimizer = optimizer,
    model = seq_model,
    loss_fn = nn.MSELoss(),
    t_u_train = Train_Data,
    t_u_val = Test_Data,
    t_c_train = Train_Outputput,
    t_c_val = Test_Output)

print()


# In[20]:


with torch.cuda.device(0):
  net = seq_model
  macs, params = get_model_complexity_info(net, (1, 5), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))


# In[21]:


##Part 2 ##


# In[22]:


new_model = nn.Sequential(
            nn.Linear(5,8),
            nn.Tanh(),
            nn.Linear(8,14),
            nn.Tanh(),
            nn.Linear(14,8),
            nn.Tanh(),
            nn.Linear(8,1))


# In[23]:


optimizer = optim.SGD(
        new_model.parameters(),
        lr=1e-2)


# In[24]:


training_loop(
    n_epochs = 3000,
    optimizer = optimizer,
    model = new_model,
    loss_fn = nn.MSELoss(),
    t_u_train = Train_Data,
    t_u_val = Test_Data,
    t_c_train = Train_Outputput,
    t_c_val = Test_Output)

print()


# In[25]:


with torch.cuda.device(0):
  net = new_model
  macs, params = get_model_complexity_info(net, (1, 5), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))


# In[26]:


## Problem 2 ##


# In[27]:


transforms.Normalize((0.4915, 0.4823, 0.4468),(0.2470, 0.2435, 0.2616))


# In[28]:


from torchvision import datasets
data_path = 'P:\STORAGE\Desktop\ML HW2'
cifar10 = datasets.CIFAR10(data_path, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4915, 0.4823, 0.4468),(0.2470, 0.2435, 0.2616))]))
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4915, 0.4823, 0.4468),(0.2470, 0.2435, 0.2616))]))


# In[29]:


model2 = nn.Sequential(
             nn.Linear(3072,512),
             nn.Tanh(),
             nn.Linear(512,10),
             nn.LogSoftmax(dim=1))
loss_fn = nn.NLLLoss()


# In[30]:


optimizer = optim.SGD(
        model2.parameters(),
        lr=1e-2)


# In[31]:


train_loader = torch.utils.data.DataLoader(cifar10, batch_size=64, shuffle=True)
n_epochs = 300


# In[33]:


for epoch in range(n_epochs):
    for imgs, labels in train_loader:
        batch_size = imgs.shape[0]
        outputs = model2(imgs.view(batch_size, -1))
        loss = loss_fn(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("Epoch: %d, Loss: %f" % (epoch, float(loss)))



# In[35]:


val_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=64, shuffle=False)

correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in val_loader:
        batch_size = imgs.shape[0]
        outputs = model2(imgs.view(batch_size, -1))
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())
print("Accuracy: %f", (correct/total))


# In[36]:


with torch.cuda.device(0):
  net = model2
  macs, params = get_model_complexity_info(net, (1, 3072), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))


# In[ ]:


## Part 2 ##


# In[37]:


model3 = nn.Sequential(
             nn.Linear(3072,512),
             nn.Tanh(),
             nn.Linear(512, 256),
             nn.Tanh(),
             nn.Linear(256, 128),
             nn.Tanh(),
             nn.Linear(128, 10),
             nn.LogSoftmax(dim=1))
loss = nn.NLLLoss()


# In[38]:


optimizer = optim.SGD(
        model3.parameters(),
        lr=1e-2)


# In[40]:


for epoch in range(n_epochs):
    for imgs, labels in train_loader:
        batch_size = imgs.shape[0]
        outputs = model3(imgs.view(batch_size, -1))
        loss = loss_fn(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("Epoch: %d, Loss: %f" % (epoch, float(loss)))



# In[41]:


val_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=64, shuffle=False)

correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in val_loader:
        batch_size = imgs.shape[0]
        outputs = model3(imgs.view(batch_size, -1))
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())
print("Accuracy: %f", correct/total)


# In[42]:


with torch.cuda.device(0):
  net = model3
  macs, params = get_model_complexity_info(net, (1, 3072), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))


# In[ ]:




