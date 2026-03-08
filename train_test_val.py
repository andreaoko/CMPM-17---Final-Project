from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
import torch
import torch.nn as nn
import torch.optim as optim
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


df = pd.read_csv("getImages2/UCSC_iNat_observations_downloads_only.csv")                                                                                          #load data into dataframe


#Augment and Show images

num_images = 100

plt.figure(figsize=(20,20))                                                                                                         #window display size of images

                                                                                                                                    #edited to pull from local files
for i in range(num_images):
    img_path = 'getImages2/' + df.iloc[i]['img_path']                                                                                     #adds additional folder logic
    name = df.iloc[i]['scientific_name']

    img = PIL.Image.open(img_path)                                                                                                  #opens from already downloaded files

    transforms = v2.Compose([
        v2.ToImage(),                                                                                                               #converts to a torch tensor image object
        v2.ToDtype(torch.float32, scale=True),                                                 
        v2.Resize((224,224)),
        v2.RandomHorizontalFlip(p=0.3),
        v2.RandomVerticalFlip(p=0.4),
        v2.ColorJitter(brightness=0.15,contrast=0.15),
        v2.RandomApply([
            v2.RandomRotation(degrees=50),
            v2.RandomResizedCrop(224, scale=(0.85,1.0)), 
        ], p=0.5),
    ])

    use_transforms = transforms(img)

    plt.subplot(10, 10, i+1)                                                                                    #plot the image in a 10 x 10 grid
    plt.imshow(v2.ToPILImage()(use_transforms))
    plt.title(name[:15], fontsize=6)
    plt.axis("off")

plt.tight_layout(pad=2, h_pad=2.5, w_pad=0.2)
plt.show()                                                                                                      #this line will block the remaining code from running.
                                                                                                                #Either close the window for the image plot or use plt.show(block=False) to skip the plot from generating                                                                                  


root = 'imagesOrganizedSplit'
                                                                                                                #Create Imagefolders
train_dataset = ImageFolder(os.path.join(root,'train'), transform=transforms)                                   #Creates a path to the respective folder
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
        self.fc1 = nn.Linear(7*7*512, 500)
        self.fc2 = nn.Linear(500, 26)
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

model = ConvNet().to(device)                                                                                  #create an instance of the model


for images, label in train_dataloader:
    print(f'image shape: {images.shape}')                                                           #print dimensions of input image shape
    out = model(images)                                                                             #pass images through test model
    print(f'output shape: {out.shape}')                                                             #print the output tnesor of model shape
    print(out[0])                                                                                   #prints image shape for first image in batch
    break
 
    
    

#image shape: torch.Size([16, 3, 500, 500])
#output shape: torch.Size([16, 26])

optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)                                                               
criterion = nn.CrossEntropyLoss()                                                                   #Define the loss function
NUM_EPOCHS = 100

for epoch in range(NUM_EPOCHS):
    model.train()

    train_correct_vals = 0
    val_correct_vals = 0
    total = 0

    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)


        train_preds = model(images)
        train_loss = criterion(train_preds, labels)

        x, t_preds = torch.max(train_preds, dim=1)                                                    #finds the highest prediction from the training predictions
        train_correct_vals += torch.sum((t_preds == labels)).item()
        total += labels.size(0)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    train_accuracy = torch.tensor(train_correct_vals / total)

    print(f"Epoch: {epoch} || Loss: {train_loss.item()} || Trainining Accuracy: {train_accuracy}")


    for images, labels in val_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        val_preds = model(images)
        val_loss = criterion(val_preds, labels)
        x, v_preds = torch.max(val_preds, dim=1)
        val_correct_vals += torch.sum((v_preds == labels)).item()

    val_accuracy = torch.tensor(val_correct_vals / len(v_preds))



model.eval()
with torch.no_grad():
    test_correct_vals = 0

    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        test_preds = model(images)
        test_loss = criterion(test_preds, labels)

        x, tt_preds = torch.max(test_preds, dim=1)
        test_correct_vals += torch.sum((tt_preds == labels).item())

    test_accuracy = torch.tensor(torch.sum(test_correct_vals) / len(tt_preds))

    print(f"Test Loss: {test_loss.item()} || Test Accuracy {test_accuracy}")
