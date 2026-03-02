from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import pandas as pd
import os
import random # for randomizing degree transformation

df = pd.read_csv("DownloadedImageData_NewPaths.csv")                                                  #place the downloaded images into a pandas dataframe
df = df.dropna() # ensures rows with missing data are dropped. 
df = df.sample(frac = 1) # mixes up the rows

transforms = v2.Compose([
    v2.ToTensor(),                                                                                              #allows images to be placed into tensors                                                                     
    v2.Resize((224,224)),                                                                                       #sets all images to a uniform size
    v2.RandomHorizontalFlip(p=0.3),                                                                             #Random image augmentations
    v2.RandomVerticalFlip(p=0.4),
    v2.RandomRotation(degrees=random.randint(1,50)),
    v2.ColorJitter(brightness=0.15,contrast=0.15),

])

root = 'imagesOrganizedSplit'

train_data = ImageFolder(os.path.join(root, 'train'), transform = transforms)
test_data = ImageFolder(os.path.join(root, 'test'), transform = transforms)
val_data = ImageFolder(os.path.join(root, 'val'), transform = transforms)

train_loader = DataLoader(train_data, batch_size = 16, shuffle = True) # This may need to be decreased if each batch is within a species folder
test_loader = DataLoader(test_data, batch_size = 8, shuffle = True)
val_loader = DataLoader(val_data, batch_size = 8, shuffle = True)