from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from PIL import Image,ImageFilter
import PIL.Image
from io import BytesIO                                      #Helps open image
import requests
import splitfolders
import os
from torchvision.datasets import ImageFolder


df = pd.read_csv("data/UCSC_iNat_observations_downloads_only.csv")

class ImageDataset(Dataset):
    
    def __init__(self, images, labels):
        super().__init__()
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
         image = self.images[index]
         tImage = self.transforms(image)
         return tImage

transforms = v2.Compose([
            v2.ToTensor(),                                                                                              #allows images to be placed into tensors                                                                     
            v2.Resize((224,224)),                                                                                       #sets all images to a uniform size
            v2.RandomHorizontalFlip(p=0.3),                                                                             #Random image augmentations
            v2.RandomVerticalFlip(p=0.4),
            v2.RandomRotation(degrees=random.randint(1,50)),
            v2.ColorJitter(brightness=0.15,contrast=0.15),
        ])
    



    
root = 'imagesOrganizedSplit'

                                                                                                            #Create Imagefolders
train_dataset = ImageFolder(os.path.join(root,'train'), transform=transforms)                               #Creates a path to the respective folder
test_dataset = ImageFolder(os.path.join(root,'test'), transform=transforms)
val_dataset = ImageFolder(os.path.join(root,'val'), transform=transforms)

                                                                                                            #Create dataloaders

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)



                                                                                                            #dataloader loops


for train_x, train_y in train_dataloader:
    print(f"Train inputs: {train_x}")
    print(f"Train outputs: {train_y}")
    break
    


for test_x, test_y in test_dataloader:
    print(f"Test inputs: {test_x}")
    print(f"Test outputs: {test_y}")
    break
    

for val_x, val_y in val_dataloader:
    print(f"Validation inputs: {val_x}")
    print(f"Validation outputs: {val_y}")
    break
    