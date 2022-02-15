#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Homework 1
#Name : Abrar Altaay
#ID: 801166376
# https://github.com/aaltaay1/Intro-to-Machine-Learning.git


# In[2]:


from torchvision import models
import torch

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt  


# In[3]:


from PIL import Image

#Red
red1 = Image.open("Red-1.jpg")
red2 = Image.open("Red-2.jpg")
red3 = Image.open("Red-3.jpg")
red4 = Image.open("Red-4.jpg")

#Green
green1 = Image.open("green-1.jpg")
green2 = Image.open("green-2.jpg")
green3 = Image.open("green-3.jpg")
green4 = Image.open("green-4.jpg")

#Blue
blue1 = Image.open("blue-1.jpg")
blue2 = Image.open("blue-2.jpg")
blue3 = Image.open("blue-3.jpg")
blue4 = Image.open("blue-4.jpg")


# In[4]:


from torchvision import transforms
transform = transforms.Compose([
    transforms.ToTensor()])

#Red
red1_t = transform(red1)
red2_t = transform(red2)
red3_t = transform(red3)
red4_t = transform(red4)

#Green
green1_t = transform(green1)
green2_t = transform(green2)
green3_t = transform(green3)
green4_t = transform(green4)

#Blue
blue1_t = transform(blue1)
blue2_t = transform(blue2)
blue3_t = transform(blue3)
blue4_t = transform(blue4)


# In[5]:


#Red Mean
red1_data = torch.mean(red1_t, dim=[1,2])
red2_data = torch.mean(red2_t, dim=[1,2])
red3_data = torch.mean(red3_t, dim=[1,2])
red4_data = torch.mean(red4_t, dim=[1,2])

#Green Mean
green1_data = torch.mean(green1_t, dim=[1,2])
green2_data = torch.mean(green2_t, dim=[1,2])
green3_data = torch.mean(green3_t, dim=[1,2])
green4_data = torch.mean(green4_t, dim=[1,2])

#Blue Mean
blue1_data = torch.mean(blue1_t, dim=[1,2])
blue2_data = torch.mean(blue2_t, dim=[1,2])
blue3_data = torch.mean(blue3_t, dim=[1,2])
blue4_data = torch.mean(blue4_t, dim=[1,2])


# In[6]:


print('Red 1 = ' , red1_data)
print('Red 2 = ' , red2_data)
print('Red 3 = ' , red3_data)
print('Red 4 = ' , red4_data)
print('\n')

print('Green 1 = ', green1_data)
print('Green 2 = ', green2_data)
print('Green 3 = ', green3_data)
print('Green 4 = ', green4_data)
print('\n')

print('Blue 1 = ', blue1_data)
print('Blue 2 = ', blue2_data)
print('Blue 3 = ', blue3_data)
print('Blue 4 = ', blue4_data)


# In[7]:


################################
################################
##########   Q 2   #############
################################
################################


# In[8]:


t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)

def model(t_u, w1, w2, b):
    return w2 * t_u ** 2 + w1 * t_u + b


# In[9]:


w1 = torch.ones(())
w2 = torch.ones(())
b = torch.zeros(())

t_p = model(t_u, w1, w2, b)
t_p


# In[10]:


def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()


# In[11]:


loss = loss_fn(t_p, t_c)
loss


# In[12]:


def dloss_fn(t_p, t_c):
    dsq_diffs = 2 * (t_p - t_c) / t_p.size(0)
    return dsq_diffs


# In[13]:


def dmodel_dw1(t_u, w1,w2,b):
    return t_u


# In[14]:


def dmodel_dw2(t_u, w1,w2,b):
    return t_u**2


# In[15]:


def dmodel_db(t_u, w1,w2,b):
    return 1.0


# In[16]:


def grad_fn(t_u, t_c, t_p, w1, w2, b):
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw1 = dloss_dtp * dmodel_dw1(t_u, w1, w2, b)
    dloss_dw2 = dloss_dtp * dmodel_dw2(t_u, w1, w2, b)
    dloss_db = dloss_dtp * dmodel_db(t_u, w1, w2, b)
    return torch.stack([dloss_dw1.sum(),dloss_dw2.sum(), dloss_db.sum()])


# In[17]:


def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        w1, w2, b = params
    
        t_p = model(t_u, w1, w2, b)
        loss = loss_fn(t_p, t_c)
        grad = grad_fn(t_u, t_c, t_p, w1, w2, b)
    
        params = params - learning_rate * grad
        
        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
    return params


# In[18]:


#Normalizing
t_un = 0.1 * t_u


# In[19]:


params = training_loop(
n_epochs = 5000,
learning_rate = .1,
params = torch.tensor([1.0, 1.0, 0.0]),
t_u = t_un,
t_c = t_c)


# In[20]:


params = training_loop(
n_epochs = 5000,
learning_rate = .01,
params = torch.tensor([1.0, 1.0, 0.0]),
t_u = t_un,
t_c = t_c)


# In[21]:


params = training_loop(
n_epochs = 5000,
learning_rate = .001,
params = torch.tensor([1.0, 1.0, 0.0]),
t_u = t_un,
t_c = t_c)


# In[22]:


params = training_loop(
n_epochs = 5000,
learning_rate = .0001,
params = torch.tensor([1.0, 1.0, 0.0]),
t_u = t_un,
t_c = t_c)


# In[23]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
t_p = model(t_un, *params)
fig = plt.figure(dpi=600)
plt.xlabel("Temperature (°Fahrenheit)")
plt.ylabel("Temperature (°Celsius)")
plt.plot(t_u.numpy(), t_p.detach().numpy())
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
plt.grid()


# In[24]:


################################
################################
##########   Q 3   #############
################################
################################


# In[25]:


import pandas as pd
#Reading training data
housing = pd.DataFrame(pd.read_csv("Housing.csv")) 
housing.head() 


# In[26]:


m = len(housing) 
m 


# In[27]:


num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price'] 
Newtrain = housing[num_vars] 
Newtrain.head() 


# In[28]:


#create tesnor
t_area = torch.tensor(Newtrain['area'])
t_bedrooms = torch.tensor(Newtrain['bedrooms'])
t_bathrooms = torch.tensor(Newtrain['bathrooms'])
t_stories = torch.tensor(Newtrain['stories'])
t_parking = torch.tensor(Newtrain['parking'])
t_price = torch.tensor(Newtrain['price'])


# In[29]:


#Normalize data
n_area = t_area / max(Newtrain['area'])
n_bedrooms = t_bedrooms / max(Newtrain['bedrooms'])
n_bathrooms = t_bathrooms / max(Newtrain['bathrooms'])
n_stories = t_stories / max(Newtrain['stories'])
n_parking = t_parking / max(Newtrain['parking'])


# In[30]:


def model(t_area, t_bedrooms, t_bathrooms, t_stories, t_parking, w1, w2, w3, w4, w5, b):
    return (w5*t_parking) + (w4*t_stories) + (w3*t_bathrooms) + (w2*t_bedrooms) + (w1*t_area) + b


# In[31]:


params = torch.tensor([1.0,1.0,1.0,1.0,1.0,0.0], requires_grad=True)


# In[32]:


params.grad is None


# In[33]:


loss = loss_fn(model(t_area, t_bedrooms, t_bathrooms, t_stories, t_parking, *params), t_price)
loss.backward()
params.grad


# In[34]:


if params.grad is not None:
    params.grad.zero_()


# In[35]:


def training_loop(n_epochs, learning_rate, params, t_area, t_bedrooms, t_bathrooms, t_stories, t_parking, t_price):
    for epoch in range(1, n_epochs + 1):
        if params.grad is not None:  # <1>
            params.grad.zero_()
        
        t_p = model(t_area, t_bedrooms, t_bathrooms, t_stories, t_parking, *params) 
        loss = loss_fn(t_p, t_price)
        loss.backward()
        
        with torch.no_grad():  # <2>
            params -= learning_rate * params.grad

        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
            
    return params


# In[36]:


params = training_loop(
    n_epochs = 5000, 
    learning_rate = 1e-1, 
    params = torch.tensor([1.0,1.0,1.0,1.0,1.0,0.0], requires_grad=True), # <1> 
    t_area = n_area,
    t_bedrooms = n_bedrooms,
    t_bathrooms = n_bathrooms,
    t_stories = n_stories,
    t_parking = n_parking,
    t_price = t_price)


# In[37]:


get_ipython().run_line_magic('matplotlib', 'inline')
#t_area, t_bedrooms, t_bathrooms, t_stories, t_parking
t_p = model(n_area, n_bedrooms, n_bathrooms, n_stories, n_parking, *params)
fig = plt.figure(dpi=600)
plt.xlabel("Price")
plt.ylabel("Houses")
plt.plot(t_p.detach().numpy())
plt.plot(t_price)


# In[ ]:




