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

df = pd.read_csv("getImages2/DownloadedImageData_NewPaths.csv")                                                                                                 #load data into dataframe

#Checking for device automatically

if torch.cuda.is_available():
    device = "cuda"
    print("CUDA is available. Using GPU.")
else:
    device = "cpu"


#Augment and Show images

num_images = 100
plt.figure(figsize=(20,20))                                                                                                         #window display size of images
                                                                                                                                    #edited to pull from local files
for i in range(num_images):
    img_path = 'getImages2/' + df.iloc[i]['img_path']                                                                                #adds additional folder logic
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
plt.show()                                                                                           #using block=False for debugging purposes


#Define datasets and dataloaders

root = 'imagesOrganizedSplit'
                                                                                                                    
train_dataset = ImageFolder(os.path.join(root,'train'), transform=transforms)                                   
test_dataset = ImageFolder(os.path.join(root,'test'), transform=transforms)
val_dataset = ImageFolder(os.path.join(root,'val'), transform=transforms)
                                                                                                    
                                                                                                                    
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)

                                                                                                                  #Check dataloader outputs 
for train_x, train_y in train_dataloader:
    print(f"\nTrain inputs: {train_x.size()}")                                                                    #Input order: ([batch size, channels, img height, img width])
    print(f"Train outputs: {train_y.size()}")                                                                     #Output order: ([batch size])
    break

for test_x, test_y in test_dataloader:
    print(f"\nTest inputs: {test_x.size()}")
    print(f"Test outputs: {test_y.size()}")
    break
    
for val_x, val_y in val_dataloader:
    print(f"\nValidation inputs: {val_x.size()}")
    print(f"Validation outputs: {val_y.size()}")
    break


#Define the CNN model class

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1 ,1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 512, 3, 1, 1)

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

model = ConvNet()   
model.to(device) 


#Check output of the model
for images, label in train_dataloader:
    print(f'\nImage shape: {images.shape}')                                                                    #print dimensions of input image shape
    output_model = model(images)                                                                             
    print(f'Output shape: {output_model.shape}')                                                             #print the output tensor of model shape
    print(output_model[0])                                                                                   #prints image shape for first image in batch
    break


#Training, Validation and Testing Loop

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)                                                               
criterion = nn.CrossEntropyLoss().to(device)  

NUM_EPOCHS = 100

#Training Loop
for epoch in range(NUM_EPOCHS):
    model.train()

    train_correct_vals = 0
    train_total_imgs = 0
    train_accuracy = 0

    v_correct_vals = 0
    v_total_imgs = 0

    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)

        train_preds = model(images)
        train_loss = criterion(train_preds, labels)

        _, tr_preds = torch.max(train_preds, dim=1)

        train_correct_vals += torch.sum((tr_preds == labels)).item()                                                       #check the correct values
        train_total_imgs += labels.size(0)


        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
    train_accuracy = train_correct_vals / train_total_imgs
    print(f"Epoch: {epoch+1}/{NUM_EPOCHS} || Training Loss: {train_loss.item()} || Training Accuracy: {train_accuracy:.6f}")

    model.eval()
    #Validation loop
    for images, labels in val_dataloader:
        images, labels = images.to(device), labels.to(device)

        val_preds = model(images)
        val_loss = criterion(val_preds, labels)

        __, v_preds = torch.max(val_preds, dim=1)
                
        v_correct_vals += torch.sum((v_preds == labels)).item()                                                       #check the correct values
        v_total_imgs += labels.size(0)        



#Testing loop
print("\n Testing Phase")

model.eval()
with torch.no_grad():
    test_correct_vals = 0
    test_total_imgs = 0

    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)

        test_preds = model(images)
        test_loss = criterion(test_preds, labels)

        __, tt_preds = torch.max(test_preds, dim=1)

        test_correct_vals += torch.sum((test_preds == labels)).item()
        test_total_imgs += labels.size(0)

    test_accuracy = test_correct_vals / test_total_imgs
    print(f"Test Loss: {test_loss.item()} || Testing Accuracy: {test_accuracy:.6f}")