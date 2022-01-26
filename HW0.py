#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Homework 0
#Name : Abrar Altaay
#ID: 801166376


# In[2]:


################################
################################
##########   Q 1   #############
################################
################################


# In[3]:


from torchvision import models


# In[4]:


resnet = models.resnet101(pretrained=True)


# In[5]:


resnet


# In[6]:


from torchvision import transforms
preprocess = transforms.Compose([
    transforms.Resize(256),      # Resize the image size to 256 pix
    transforms.CenterCrop(224),  # crop the image
    transforms.ToTensor(),       # send to tensor 
    transforms.Normalize(    
        mean=[0.485, 0.456, 0.406], # the mean
        std=[0.229, 0.224, 0.225]   
)])


# In[7]:


from PIL import Image
img1  = Image.open("Airplane.jpg")


# In[8]:


img2 = Image.open("Car.jpg")


# In[9]:


img3 = Image.open("Cat.jpg")


# In[10]:


img4 = Image.open("Chicken.jpg")


# In[11]:


img5 = Image.open("Sky.jpg")


# In[12]:


img_t1 = preprocess(img1)
img_t2 = preprocess(img2)
img_t3 = preprocess(img3)
img_t4 = preprocess(img4)
img_t5 = preprocess(img5)


# In[13]:


import torch
batch_t1 = torch.unsqueeze(img_t1, 0)
batch_t2 = torch.unsqueeze(img_t2, 0)
batch_t3 = torch.unsqueeze(img_t3, 0)
batch_t4 = torch.unsqueeze(img_t4, 0)
batch_t5 = torch.unsqueeze(img_t5, 0)


# In[14]:


resnet.eval()


# In[15]:


out1 = resnet(batch_t1)
out2 = resnet(batch_t2)
out3 = resnet(batch_t3)
out4 = resnet(batch_t4)
out5 = resnet(batch_t5)


# In[16]:


with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]


# In[17]:


_, index1 = torch.max(out1, 1)
_, index2 = torch.max(out2, 1)
_, index3 = torch.max(out3, 1)
_, index4 = torch.max(out4, 1)
_, index5 = torch.max(out5, 1)


# In[18]:


percentage1 = torch.nn.functional.softmax(out1, dim=1)[0] * 100
percentage2 = torch.nn.functional.softmax(out2, dim=1)[0] * 100
percentage3 = torch.nn.functional.softmax(out3, dim=1)[0] * 100
percentage4 = torch.nn.functional.softmax(out4, dim=1)[0] * 100
percentage5 = torch.nn.functional.softmax(out5, dim=1)[0] * 100


# In[19]:


#Airplane
_, indices = torch.sort(out1, descending=True)
[(labels[idx], percentage1[idx].item()) for idx in indices[0][:5]]


# In[20]:


# Seems to have guessed correctly what the image is, 
#there was also a 5% chance it's a wing that is correctly
#would have been cool to see what type of airplane it was
#such as a pasanger plane, or brand.
img1


# In[21]:


#Car
_, indices2 = torch.sort(out2, descending=True)
[(labels[idx], percentage2[idx].item()) for idx in indices2[0][:5]]


# In[22]:


img2
#Resnet seems to have gotten this incorrectly.
# It was not able to determine this was a sedan, and Hyundai  at that
#With clear indication it was a Hyundai from the logo  


# In[23]:


#Cat
_, indices3 = torch.sort(out3, descending=True)
[(labels[idx], percentage3[idx].item()) for idx in indices3[0][:5]]


# In[70]:


img3
# Renet seemed to have gotten this correctly with only a 79% of a tiger cat
# this is in fact a tiger cat with 20% a tabby cat, but a tiger cat is a tabby cat
# so this is accutaly pretty accurate! 


# In[24]:


#Chicken
_, indices4 = torch.sort(out4, descending=True)
[(labels[idx], percentage4[idx].item()) for idx in indices4[0][:5]]


# In[71]:


img4
# resnet got this almost 100% accuarte with a 97% accuracy that this is a hen
# a hen is different from a cock as a hen is a female chicken which is correct


# In[25]:


#Sky
_, indices5 = torch.sort(out5, descending=True)
[(labels[idx], percentage5[idx].item()) for idx in indices5[0][:5]]


# In[72]:


img5
# Resnet got this very wrong. It said this is a semi/truck which is very wrong
# this is just an image of the sky -- I believe this is because that the resnet 
# was trained on a lot of images that include trucks, and trucks are always outside probably with 
# nice weather and sky such as this, so the closest thing it has to compare to is 'semis'


# In[26]:


################################
################################
##########   Q 2   #############
################################
################################


# In[27]:


import torch
import torch.nn as nn

class ResNetBlock(nn.Module): # <1>

    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        conv_block = []

        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       nn.InstanceNorm2d(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x) # <2>
        return out


class ResNetGenerator(nn.Module):

    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9): # <3> 

        assert(n_blocks >= 0)
        super(ResNetGenerator, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=True),
                      nn.InstanceNorm2d(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResNetBlock(ngf * mult)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=True),
                      nn.InstanceNorm2d(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input): # <3>
        return self.model(input)


# In[28]:


netG = ResNetGenerator()


# In[29]:


model_path = 'horse2zebra_0.4.0.pth'
model_data = torch.load(model_path)
netG.load_state_dict(model_data)


# In[30]:


netG.eval()


# In[31]:


from PIL import Image
from torchvision import transforms


# In[32]:


preprocess = transforms.Compose([transforms.Resize(256),
                                 transforms.ToTensor()])


# In[33]:


img11 = Image.open("horse1.jpg")
img22 = Image.open("horse2.jpg")
img33 = Image.open("horse3.jpg")
img44 = Image.open("horse4.jpg")
img55 = Image.open("horse5.jpg")


# In[34]:


img_t1 = preprocess(img11)
img_t2 = preprocess(img22)
img_t3 = preprocess(img33)
img_t4 = preprocess(img44)
img_t5 = preprocess(img55)


# In[35]:


batch_t1 = torch.unsqueeze(img_t1, 0)
batch_t2 = torch.unsqueeze(img_t2, 0)
batch_t3 = torch.unsqueeze(img_t3, 0)
batch_t4 = torch.unsqueeze(img_t4, 0)
batch_t5 = torch.unsqueeze(img_t5, 0)


# In[36]:



batch_out1 = netG(batch_t1)
batch_out2 = netG(batch_t2)
batch_out3 = netG(batch_t3)
batch_out4 = netG(batch_t4)
batch_out5 = netG(batch_t5)


# In[63]:


img11


# In[62]:


out_t1 = (batch_out1.data.squeeze() + 1.0) / 2.0
out_img1 = transforms.ToPILImage()(out_t1)
out_img1

#The output seems to have miss catagorized the red colors such as the 
#bin and the roses were turned into a zebra pattern also
#this could be becuase the resnet is trained on looking for simialar colors 
#and the red happened to match


# In[65]:


img22


# In[38]:


out_t2 = (batch_out2.data.squeeze() + 1.0) / 2.0
out_img2 = transforms.ToPILImage()(out_t2)
out_img2
#This image looked slightly better than the last one
#this was interesting that it got it correct as the horse was a game model
#and not a real horse -- seems to have miss understood the a corner of the wheel
# and gave the wheel a zebra pattern also


# In[66]:


img33


# In[39]:


out_t3 = (batch_out3.data.squeeze() + 1.0) / 2.0
out_img3 = transforms.ToPILImage()(out_t3)
out_img3

#This image was very high quality with a lot of different colors
# it seemed to not understand black horses as wheel as brown horses
#this could be becuase the model was not trained on many black or light brown horses
# and was primarily trained on brown horses


# In[67]:


img44


# In[40]:


out_t4 = (batch_out4.data.squeeze() + 1.0) / 2.0
out_img4 = transforms.ToPILImage()(out_t4)
out_img4

# there's 3 horses in the mage, but one of the horses was covered by another
#horse, the other horse seem to have been *deleted* and only two horses were clored in


# In[68]:


img55


# In[41]:


out_t5 = (batch_out5.data.squeeze() + 1.0) / 2.0
out_img5 = transforms.ToPILImage()(out_t5)
out_img5
#this was my faviorte picture to try as the horse is moving at a weird angle
# it was interesting to see how the resnet would classify it and it did a fantastic job
#the image does seem to be a bit miss toned but it worked perfectly around the edges and hair


# In[42]:


################################
################################
##########   Q 3   #############
################################
################################


# In[43]:


# For question 1

import torchvision.models as models
import torch
from ptflops import get_model_complexity_info

with torch.cuda.device(0):
  net = models.resnet101()
  macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))


# In[44]:


# For question 2 

with torch.cuda.device(0):
  net = netG
  macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))


# In[ ]:


#Resnet was trained on over 45 million over parameters since it has a way bigger number of parameters of 44.55 M 
# Resnet did not need to do a lot of computational complexist as long as it was given an explicit set of parameter.

# As for resgen, ti was trained on only pictures of horse and zebra
# but needed to do as lot more computations to figure out where the horse is
# and how to color the horse correctly. 


# In[45]:


################################
################################
##########   Q 4   #############
################################
################################


# In[46]:


import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model.eval()


# In[47]:


import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)


# In[48]:


from PIL import Image
from torchvision import transforms


# In[49]:


#input the images

input_image1 = img1
input_image2 = img2
input_image3 = img3
input_image4 = img4
input_image5 = img5


# In[50]:


# function process of normalzing the images
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# In[51]:


# run function
img_t1 = preprocess(input_image1)
img_t2 = preprocess(input_image2)
img_t3 = preprocess(input_image3)
img_t4 = preprocess(input_image4)
img_t5 = preprocess(input_image5)


# In[52]:


#tensor process of all 5 image from problem 1


###### IMAGE 1
input_tensor1 = preprocess(input_image1)
input_batch1 = input_tensor1.unsqueeze(0) 

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch1 = input_batch1.to('cuda')
    model.to('cuda')
    
with torch.no_grad():
    output1 = model(input_batch1)

# The output has unnormalized scores. To get probabilities, you can run a softmax on it.    
probabilities1 = torch.nn.functional.softmax(output1[0], dim=0)

###### IMAGE 2
    
input_tensor2 = preprocess(input_image2)
input_batch2 = input_tensor2.unsqueeze(0) 
if torch.cuda.is_available():
    input_batch2 = input_batch2.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output2 = model(input_batch2)


probabilities2 = torch.nn.functional.softmax(output2[0], dim=0)

###### IMAGE 3

input_tensor3 = preprocess(input_image3)
input_batch3 = input_tensor3.unsqueeze(0) 
if torch.cuda.is_available():
    input_batch3 = input_batch3.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output3 = model(input_batch3)
    
probabilities3 = torch.nn.functional.softmax(output3[0], dim=0)
    
###### IMAGE 4
    
input_tensor4 = preprocess(input_image4)
input_batch4 = input_tensor4.unsqueeze(0) 
if torch.cuda.is_available():
    input_batch4 = input_batch4.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output4 = model(input_batch4)

probabilities4 = torch.nn.functional.softmax(output4[0], dim=0)

###### IMAGE 5

input_tensor5 = preprocess(input_image5)
input_batch5 = input_tensor5.unsqueeze(0) 
if torch.cuda.is_available():
    input_batch5 = input_batch5.to('cuda')
    model.to('cuda')
with torch.no_grad():
    output5 = model(input_batch5)

probabilities5 = torch.nn.functional.softmax(output5[0], dim=0)


# In[53]:


# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print("Output for img 1")
print(output1[0])
print("\n\n\n\n")

print("Output for img 2")
print(output2[0])
print("\n\n\n\n")

print("Output for img 3")
print(output3[0])
print("\n\n\n\n")

print("Output for img 4")
print(output4[0])
print("\n\n\n\n")

print("Output for img 5")
print(output5[0])
print("\n\n\n\n")


# In[54]:


# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print("Output for img 1")
print(probabilities1)
print("\n\n\n\n")

print("Output for img 2")
print(probabilities2)
print("\n\n\n\n")

print("Output for img 3")
print(probabilities3)
print("\n\n\n\n")

print("Output for img 4")
print(probabilities4)
print("\n\n\n\n")

print("Output for img 5")
print(probabilities5)
print("\n\n\n\n")


# In[74]:


# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Show top categories per image
print("Image 1 - Airplane")
top5_prob1, top5_catid1 = torch.topk(probabilities1, 5)
for i in range(top5_prob1.size(0)):
    print(categories[top5_catid1[i]], top5_prob1[i].item())
print("\n\n")


print("Image 2 - Car")
top5_prob2, top5_catid2 = torch.topk(probabilities2, 5)
for i in range(top5_prob2.size(0)):
    print(categories[top5_catid2[i]], top5_prob2[i].item())
print("\n\n")

print("Image 3 - Cat")
top5_prob3, top5_catid3 = torch.topk(probabilities3, 5)
for i in range(top5_prob3.size(0)):
    print(categories[top5_catid3[i]], top5_prob3[i].item())

    
print("Image 4 - Chicken")
print("\n\n")
top5_prob4, top5_catid4 = torch.topk(probabilities4, 5)
for i in range(top5_prob4.size(0)):
    print(categories[top5_catid4[i]], top5_prob4[i].item())


print("Image 5 - sky")
print("\n\n")
top5_prob5, top5_catid5 = torch.topk(probabilities5, 5)
for i in range(top5_prob5.size(0)):
    print(categories[top5_catid5[i]], top5_prob5[i].item())


# In[56]:


# For question 4 part 2

with torch.cuda.device(0):
  net = models.mobilenet_v2()
  macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))


# In[ ]:


# when comparing Resnet to mobilenet v2 we can see that the accuracy of mobilenet v2 decreased 
# by 10% +- as compared -- on the upside, the computational complexity and the parameters decreased.  
# the resent complexity was 7.85 GMac with a 44.55 M parameters 
# with this, we reduced the complexity by a factor of 25 times!
# and reduced the parameters by 12 times for a faster result, but a less accurate model. 

# Resnet
#Computational complexity:       7.85 GMac
#Number of parameters:           44.55 M 

