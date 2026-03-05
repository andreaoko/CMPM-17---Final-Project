from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image,ImageFilter
import PIL.Image
import os
from torchvision.datasets import ImageFolder

df = pd.read_csv("data/UCSC_iNat_observations_downloads_only.csv")                                                                                          #load data into dataframe


#Augment and Show images

num_images = 100

plt.figure(figsize=(20,20))                                                                                                         #window display size of images

                                                                                                                                    #edited to pull from local files
for i in range(num_images):
    img_path = 'data/' + df.iloc[i]['img_path']                                                                                     #adds additional folder logic
    name = df.iloc[i]['scientific_name']

    img = PIL.Image.open(img_path)                                                                                                  #opens from already downloaded files

    transforms = v2.Compose([
        v2.ToImage(),                                                                                                               #converts to a torch tensor image object
        v2.ToDtype(torch.float32, scale=True),                                                 
        v2.Resize((500,500)),
        v2.RandomHorizontalFlip(p=0.3),
        v2.RandomVerticalFlip(p=0.4),
        v2.ColorJitter(brightness=0.15,contrast=0.15),
        v2.RandomApply([
            v2.RandomRotation(degrees=50),
            v2.RandomResizedCrop(500, scale=(0.85,1.0)), 
        ], p=0.5),
    ])

    use_transforms = transforms(img)

    plt.subplot(10, 10, i+1)                                                                                                        #plot the image in a 10 x 10 grid
    plt.imshow(v2.ToPILImage()(use_transforms))
    plt.title(name[:15], fontsize=6)
    plt.axis("off")

plt.tight_layout(pad=2, h_pad=2.5, w_pad=0.2)
plt.show()


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


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 1 ,1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1 )
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1 )
        self.conv5 = nn.Conv2d(256, 512, 3, 1,1 )

        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(15*15*512, 500)
        self.fc2 = nn.Linear(400, 26)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, X):
        X = self.pool(self.relu(self.conv1(X)))
        X = self.pool(self.relu(self.conv2(X)))
        X = self.pool(self.relu(self.conv3(X)))
        X = self.pool(self.relu(self.conv4(X)))
        X = self.pool(self.relu(self.conv5(X)))
        X = X.flatten(start_dim=1)
        X = self.relu(self.fc1(X))
        output = self.fc2(X)
        return output

test_model = ConvNet()


for x, y in train_dataloader:
    print(f'image shape: {x.shape}')
    out = test_model(x)
    print(f'output shape: {out.shape}')
    print(out[0])
    break



