from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
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
#print(df['image_url'])                                                                                              #access all values in image column
all_urls = list(df['image_url'])
url_df = df['image_url']
print(all_urls)

N = 100
rows = 10
cols = 10


for i, (idx, row) in enumerate(df.iterrows()):                                                                  #idc is the dataframe index value, row is row value
    
    if i >= N:
        break
    else:
        url = row['image_url']
        scientific_name = row['scientific_name']

        result = requests.get(url, timeout=60)
        image = PIL.Image.open(BytesIO(result.content))

        plt.subplot(rows, cols, i+1 ) 
        plt.imshow(image)
        plt.axis('off') # Hide axes
        plt.title(scientific_name[:20])

plt.tight_layout() # this line of code makes the layout/format nice with even spacing
plt.show()























#url = df.iloc[5]['image_url']
#print(url)



