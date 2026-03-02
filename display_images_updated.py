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

df = pd.read_csv("data/UCSC_iNat_observations_downloads_only.csv")                                                  #place the downloaded images into a pandas dataframe

num_images = 100

plt.figure(figsize=(24,24))                                                                                         #window display size of images

for i in range(num_images):
    url = df.iloc[i]['image_url']                                                                                   #locate the image url; can be changed later to get the img_path which is much faster for computation
    name = df.iloc[i]['common_name']                                                                                #locates common name of plant
    sci_name = df.iloc[i]['scientific_name']                                                                        #locates scientific name of plant

    result = requests.get(url, timeout=15)                                                                          #request the image data from the url
    img = PIL.Image.open(BytesIO(result.content))                                                                   #opens the image

    transforms = v2.Compose([
        v2.ToTensor(),                                                                                              #allows images to be placed into tensors                                                                     
        v2.Resize((224,224)),                                                                                       #sets all images to a uniform size
        v2.RandomHorizontalFlip(p=0.3),                                                                             #Random image augmentations
        v2.RandomVerticalFlip(p=0.4),
        v2.RandomRotation(degrees=random.randint(1,50)),
        v2.ColorJitter(brightness=0.15,contrast=0.15),

    ])


    use_transforms = transforms(img)                                                                                #implement the transformations made

    plt.subplot(10, 10, i+1)
    plt.imshow(v2.ToPILImage()(use_transforms))                                                                     #allows images to be shown after being converted to a tensor
    plt.title(name[:15], fontsize=6)                                                                                #displays part of plant names and makes the font smaller
    plt.subplots_adjust(hspace=0.9, wspace=0.3)                                                                     #fixes spacing between images both horizontally and width

    plt.axis("off")

plt.tight_layout() # this line of code makes the layout/format nice with even spacing
plt.show()





