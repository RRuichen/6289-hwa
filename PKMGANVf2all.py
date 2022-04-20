# -*- coding: utf-8 -*-

#https://blog.jovian.ai/pokegan-generating-fake-pokemon-with-a-generative-adversarial-network-f540db81548d
#https://jovian.ai/jkleiber8/course-project-pokegan/v/3?utm_source=embed





#manually select and tag pokemon picture 


#Tripling the dataset is not good for coloring



#png to jpg  64*64 


import glob
from PIL import Image
import os.path
def selectcolorpic(pngpic,width=64,height=64):
    img=Image.open(pngpic)
    try:
        new_img=img.resize((width,height),Image.BILINEAR)
        outdir="~\\pokemon\\color\\sugimori64pn1g"
        new_img.save(os.path.join(outdir,os.path.basename(pngpic)))
    except Exception as e:
        print(e)


for pngpic in glob.glob("~\\pokemon\\color\\sugimori\\*.png"):
    selectcolorpic(pngpic)
    
    
 
from PIL import Image
import glob
import os
import matplotlib.pyplot as plt

pngpath=str("~\\pokemon\\color\\sugimori64png")
jpgpath=str("~\\pokemon\\color\\sugimori64jpg")

for pngpic in glob.glob(pngpath+"\\*.png"):
    img=Image.open(pngpic)
    try:
        file_name = pngpic.replace('png', 'jpg')
        #outdircolor=str(get_color(frame))
        #outdir="~\\pokemon\\color\\sugimori64jpg"
        img=img.convert("RGB")
        img.save(file_name) 
    except Exception as e:
        print(e)

jpgpic=str("~\\pokemon\\color\\sugimori64jpg\\11.jpg")
img=Image.open(jpgpic)





#colorlist

import numpy as np
import collections
 
#indentify color clusster 
#example：{"colorname": [minRBG, maxRBG]}
#{'red': [array([160, 43, 46]), array([179, 255, 255])]}
 
def getColorList():
  dict = collections.defaultdict(list)
 
  # black
  lower_black = np.array([5, 5, 5])
  upper_black = np.array([180, 254, 46])
  color_list = []
  color_list.append(lower_black)
  color_list.append(upper_black)
  dict['black'] = color_list
 
  # # gray
  # lower_gray = np.array([0, 0, 46])
  # upper_gray = np.array([180, 43, 220])
  # color_list = []
  # color_list.append(lower_gray)
  # color_list.append(upper_gray)
  # dict['gray']=color_list
 
  # white
  lower_white = np.array([1, 1, 221])
  upper_white = np.array([180, 30, 254])
  color_list = []
  color_list.append(lower_white)
  color_list.append(upper_white)
  dict['white'] = color_list
 
  # red
  lower_red = np.array([156, 43, 46])
  upper_red = np.array([180, 255, 255])
  color_list = []
  color_list.append(lower_red)
  color_list.append(upper_red)
  dict['red']=color_list
 
  # red2
  lower_red = np.array([0, 43, 46])
  upper_red = np.array([10, 255, 255])
  color_list = []
  color_list.append(lower_red)
  color_list.append(upper_red)
  dict['red2'] = color_list
 
  # orange
  lower_orange = np.array([11, 43, 46])
  upper_orange = np.array([25, 255, 255])
  color_list = []
  color_list.append(lower_orange)
  color_list.append(upper_orange)
  dict['orange'] = color_list
 
  # yellow
  lower_yellow = np.array([26, 43, 46])
  upper_yellow = np.array([34, 255, 255])
  color_list = []
  color_list.append(lower_yellow)
  color_list.append(upper_yellow)
  dict['yellow'] = color_list
 
  # green
  lower_green = np.array([35, 43, 46])
  upper_green = np.array([77, 255, 255])
  color_list = []
  color_list.append(lower_green)
  color_list.append(upper_green)
  dict['green'] = color_list
 
  # cyan
  lower_cyan = np.array([78, 43, 46])
  upper_cyan = np.array([99, 255, 255])
  color_list = []
  color_list.append(lower_cyan)
  color_list.append(upper_cyan)
  dict['cyan'] = color_list
 
  # blue
  lower_blue = np.array([100, 43, 46])
  upper_blue = np.array([124, 255, 255])
  color_list = []
  color_list.append(lower_blue)
  color_list.append(upper_blue)
  dict['blue'] = color_list
 
  # purple
  lower_purple = np.array([125, 43, 46])
  upper_purple = np.array([155, 255, 255])
  color_list = []
  color_list.append(lower_purple)
  color_list.append(upper_purple)
  dict['purple'] = color_list
 
  return dict
 
 
if __name__ == '__main__':
  color_dict = getColorList()
  print(color_dict)
 
  num = len(color_dict)
  print('num=',num)
 
  for d in color_dict:
    print('key=',d)
    print('value=',color_dict[d][1])

# color calculation

import cv2
import numpy as np
import colorList
#import tkcolorList
 
filename='~\\pokemon\\'
 
#image process
def get_color(frame):
  #print('go in get_color')
  hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
  maxsum = -100
  color = None
  #color_dict = colorList.getColorList()
  color_dict = colorList.getColorList()
  for d in color_dict:
    mask = cv2.inRange(hsv,color_dict[d][0],color_dict[d][1])
    cv2.imwrite(d+'.jpg',mask)
    binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    binary = cv2.dilate(binary,None,iterations=2)
    #img, cnts, hiera = cv2.findContours(binary.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts, hiera = cv2.findContours(binary.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    sum = 0
    for c in cnts:
      sum+=cv2.contourArea(c)
    if sum > maxsum :
      maxsum = sum
      color = d
 
  return color
 
 
if __name__ == '__main__':
  frame = cv2.imread(filename)
  print(get_color(frame))


#color classify

from PIL import Image
import glob
import os
import matplotlib.pyplot as plt

pngpath=str("~\\pokemon\\color\\sugimori64png")
jpgpath=str("~\\pokemon\\color\\sugimori64jpg")

for jpgpic in glob.glob(jpgpath+"\\*.jpg"):
    img=Image.open(jpgpic)
    frame = cv2.imread(jpgpic)
    try:
        outdircolor=str(get_color(frame))
        outdir="~\\pokemon\\color\\colorselect\\"+outdircolor
        img.save(os.path.join(outdir,os.path.basename(jpgpic)))
    except Exception as e:
        print(e)













#Data Processing





#tripled the size of training data

from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import os
import torch
import torch.utils.data as data
from PIL import Image
#stat628910hw02
#g22663921
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path

import os
os.getcwd()

#
#setting
IMAGE_DIR = "~\\pokemon\\color\\colorselect"
#IMAGE_DIR = "~\\pokemon\\color\\colorselect\\redz"
#IMAGE_DIR = "~\\pokemon\\color\\colorselect\\bluez"
#IMAGE_DIR = "~\\pokemon\\v1\\ccon"
RESULTS_DIR = "~\\pokemon\\color\\result"
#RESULTS_DIR = "~\\pokemon\\color\\redz"
#RESULTS_DIR = "~\\pokemon\\color\\bluez"
image_size = 64#64
batch_size = 8#8
normalization_stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) # Convert channels from [0, 1] to [-1, 1]


#myImageFloder1=myImageFloder(root = "~\\makedata7", label = "~\\makedata7\\flowerz7_train.txt", transform = mytransform)

#Tripling the dataset size in order to get better results
normal_dataset = ImageFolder(IMAGE_DIR, transform=T.Compose([
    T.Resize(image_size),
    T.CenterCrop(image_size),
    T.ToTensor(),
    T.Normalize(*normalization_stats)]))

# Augment the dataset with mirrored images
mirror_dataset = ImageFolder(IMAGE_DIR, transform=T.Compose([
    T.Resize(image_size),
    T.CenterCrop(image_size),
    T.RandomHorizontalFlip(p=1.0),
    T.ToTensor(),
    T.Normalize(*normalization_stats)]))

# Augment the dataset with color changes
color_jitter_dataset = ImageFolder(IMAGE_DIR, transform=T.Compose([
    T.Resize(image_size),
    T.CenterCrop(image_size),
    T.ColorJitter(0.5, 0.5, 0.5),
    T.ToTensor(),
    T.Normalize(*normalization_stats)]))

classes1 = ('01NORMAL', '02FIRE', '03WATER', '04ELECTRIC', '05GRASS', '06ICE', '07FIGHTING', '08POISON', '09GROUND', '10FLYING', '11PSYCHIC', '12BUG', '13ROCK', '14GHOST', '15DRAGON', '16DARK', '17STEEL', '18FAIRY')
classes2 = ('black','blue','cyan','green','orange','purple','red','red2','white','yellow')

# Combine the datasets
#dataset_list = [normal_dataset, mirror_dataset, color_jitter_dataset]
dataset_list = [normal_dataset, mirror_dataset]
#dataset_list = [normal_dataset]


dataset = ConcatDataset(dataset_list)

#dataset = normal_dataset











dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=False)


#Since these images have been normalized to [-1, 1], we need to denormalize them in order to view them. Below is a denormalization function to do just that.
def denorm(image):
    return image * normalization_stats[1][0] + normalization_stats[0][0]

#Now let's show a sample batch of real Pokemon images

import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
#%matplotlib inline

def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))
    
def show_batch(dataloader, nmax=64):
    for images, _ in dataloader:
        show_images(images, nmax)
        break

#show_batch(dataloader)
#show_batch(mirror_dataset)
#show_batch(color_jitter_dataset)


#
#Discriminator Models
import torch.nn as nn
disc_64_2 = nn.Sequential(
    # Input is 3 x 64 x 64
    nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2, inplace=True),
    # Layer Output: 64 x 32 x 32
    
    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),
    # Layer Output: 128 x 16 x 16
    
    nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),
    # Layer Output: 128 x 8 x 8
    
    nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),
    # Layer Output: 128 x 4 x 4
    
    # With a 4x4, we can condense the channels into a 1 x 1 x 1 to produce output
    nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0, bias=False),
    nn.Flatten(),
    nn.Sigmoid()
)

#
#Generator Models

seed_size = 16

gen_64_2 = nn.Sequential(
    # Input seed_size x 1 x 1
    nn.ConvTranspose2d(seed_size, 128, kernel_size=4, padding=0, stride=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    # Layer output: 256 x 4 x 4
    
    nn.ConvTranspose2d(128, 128, kernel_size=4, padding=1, stride=2, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    # Layer output: 128 x 8 x 8
    
    nn.ConvTranspose2d(128, 128, kernel_size=4, padding=1, stride=2, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    # Layer output: 64 x 16 x 16
    
    nn.ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=2, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    # Layer output: 32 x 32 x 32
    
    nn.ConvTranspose2d(64, 3, kernel_size=4, padding=1, stride=2, bias=False),
    nn.Tanh()
    # Output: 3 x 64 x 64
)

#
#Model Testing


test_model_size = False

if test_model_size:
    # Make some latent tensors to seed the generator
    seed_batch = torch.randn(batch_size, seed_size, 1, 1, device=device)

    # Get some fake pokemon
    generator=gen_64_1
    to_device(generator, device)
    fake_pokemon = generator(seed_batch)
    print(fake_pokemon.size())


#
#GPU Setup

def get_training_device():
    # Use the GPU if possible
    if torch.cuda.is_available():
        return torch.device('cuda')
    # Otherwise use the CPU :-(
    return torch.device('cpu')

def to_device(data, device):
    # This moves the tensors to the device (GPU, CPU)
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dataloader: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dataloader)
    


device = get_training_device()
device

#device(type='cuda')
#If the above output doesn't say something about 'cuda', 
#then make sure the notebook is set up to run on the GPU accelerator.
#device(type='cpu')


# Using the dataloader from the top of the notebook, and the selected device
# create a device data loader
dev_dataloader = DeviceDataLoader(dataloader, device)



#
#Training Functions
#

#Discriminator Training Functions
def train_discriminator(real_pokemon, disc_optimizer):
    # Reset the gradients for the optimizer
    disc_optimizer.zero_grad()
    
    # Train on the real images
    real_predictions = discriminator(real_pokemon)
    # real_targets = torch.zeros(real_pokemon.size(0), 1, device=device) # All of these are real, so the target is 0.
    real_targets = torch.rand(real_pokemon.size(0), 1, device=device) * (0.1 - 0) + 0 # Add some noisy labels to make the discriminator think harder.
    real_loss = F.binary_cross_entropy(real_predictions, real_targets) # Can do binary loss function because it is a binary classifier
    real_score = torch.mean(real_predictions).item() # How well does the discriminator classify the real pokemon? (Higher score is better for the discriminator)
    
    # Make some latent tensors to seed the generator
    latent_batch = torch.randn(batch_size, seed_size, 1, 1, device=device)
    
    # Get some fake pokemon
    fake_pokemon = generator(latent_batch)
    
    # Train on the generator's current efforts to trick the discriminator
    gen_predictions = discriminator(fake_pokemon)
    # gen_targets = torch.ones(fake_pokemon.size(0), 1, device=device)
    gen_targets = torch.rand(fake_pokemon.size(0), 1, device=device) * (1 - 0.9) + 0.9 # Add some noisy labels to make the discriminator think harder.
    gen_loss = F.binary_cross_entropy(gen_predictions, gen_targets)
    gen_score = torch.mean(gen_predictions).item() # How well did the discriminator classify the fake pokemon? (Lower score is better for the discriminator)
    
    # Update the discriminator weights
    total_loss = real_loss + gen_loss
    total_loss.backward()
    disc_optimizer.step()
    return total_loss.item(), real_score, gen_score


#Generator Training Functions
def train_generator(gen_optimizer):
    # Clear the generator gradients
    gen_optimizer.zero_grad()
    
    # Generate some fake pokemon
    latent_batch = torch.randn(batch_size, seed_size, 1, 1, device=device)
    fake_pokemon = generator(latent_batch)
    
    # Test against the discriminator
    disc_predictions = discriminator(fake_pokemon)
    targets = torch.zeros(fake_pokemon.size(0), 1, device=device) # We want the discriminator to think these images are real.
    loss = F.binary_cross_entropy(disc_predictions, targets) # How well did the generator do? (How much did the discriminator believe the generator?)
    
    # Update the generator based on how well it fooled the discriminator
    loss.backward()
    gen_optimizer.step()
    
    # Return generator loss
    return loss.item()

#
#Results Viewer
#


import os
from torchvision.utils import save_image

#RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)



def save_results(index, latent_batch, show=True):
    # Generate fake pokemon
    fake_pokemon = generator(latent_batch)
    
    # Make the filename for the output
    fake_file = "result-image-{0:0=4d}.png".format(index)
    
    # Save the image
    save_image(denorm(fake_pokemon), os.path.join(RESULTS_DIR, fake_file), nrow=8)
    print("Result Saved!")
    
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_pokemon.cpu().detach(), nrow=8).permute(1, 2, 0))




#Full Training

from tqdm.notebook import tqdm
import torch.nn.functional as F

# Static generation seed batch
fixed_latent_batch = torch.randn(64, seed_size, 1, 1, device=device)

def train(epochs, learning_rate, start_idx=1):
    # Empty the GPU cache to save some memory
    torch.cuda.empty_cache()
    
    # Track losses and scores
    disc_losses = []
    disc_scores = []
    gen_losses = []
    gen_scores = []
    
    # Create the optimizers
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.9))
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.9))
    
    # Run the loop
    for epoch in range(epochs):
        # Go through each image
        for real_img, _ in tqdm(dev_dataloader):
            # Train the discriminator
            disc_loss, real_score, gen_score = train_discriminator(real_img, disc_optimizer)

            # Train the generator
            gen_loss = train_generator(gen_optimizer)
        
        # Collect results
        disc_losses.append(disc_loss)
        disc_scores.append(real_score)
        gen_losses.append(gen_loss)
        gen_scores.append(gen_score)
        
        # Print the losses and scores
        print("Epoch [{}/{}], gen_loss: {:.4f}, disc_loss: {:.4f}, real_score: {:.4f}, gen_score: {:.4f}".format(
            epoch+start_idx, epochs, gen_loss, disc_loss, real_score, gen_score))
        
        # Save the images and show the progress
        save_results(epoch + start_idx, fixed_latent_batch, show=False)
    
    # Return stats
    return disc_losses, disc_scores, gen_losses, gen_scores





#GPU Clean-up

device = get_training_device()
device


def debug_memory():
    import collections, gc, resource, torch
    print('maxrss = {}'.format(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape))
                                  for o in gc.get_objects()
                                  if torch.is_tensor(o))
    for line in tensors.items():
        print('{}\t{}'.format(*line))


mem_debug = False
if mem_debug:
    debug_memory()

# Clean up everything
cleanup = False
if cleanup:
    import gc
    del dev_dataloader
    del discriminator
    del generator
    dev_dataloader = None
    discriminator = None
    generator = None
    gc.collect()
    torch.cuda.empty_cache()

# Re-initialize the device dataloader
dev_dataloader = DeviceDataLoader(dataloader, device)






#
#Selecting Models
#

# Discriminators
# discriminator = disc_1
# discriminator = disc_2
# discriminator = disc_3
# discriminator = disc_5

# 64 x 64 Discriminators
# discriminator = disc_64_1
discriminator = disc_64_2

# Send to device
discriminator = to_device(discriminator, device)

# Generators
# generator = gen_1
# generator = gen_3
# generator = gen_5

# 64 x 64 Generators
# generator = gen_64_1
generator = gen_64_2

# Send to device
generator = to_device(generator, device)

#Training Time

# learning_rate = 0.0025 # worked fairly well for disc/gen_64_1
learning_rate = 0.0025#0.00275
epochs = 450#50


history = train(epochs, learning_rate)


#
#View Results
#

from IPython.display import Image

#Image('./results/result-image-0001.png')
#Image('./results/result-image-0010.png')
#Image('./results/result-image-0025.png')
#Image('./results/result-image-0050.png')






#
#Performance Analysis
#


# Extract metrics
disc_losses, disc_scores, gen_losses, gen_scores = history

# Plot generator and discriminator losses
plt.plot(disc_losses, '-')
plt.plot(gen_losses, '-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Discriminator', 'Generator'])
plt.title('Losses');


# Plots scores vs. epochs
plt.plot(disc_scores, '-')
plt.plot(gen_scores, '-')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend(['Real', 'Fake'])
plt.title('Scores');



