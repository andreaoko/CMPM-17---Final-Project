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
import time


df = pd.read_csv("DownloadedImageData_NewPaths.csv")                                                                                                 #load data into dataframe


#Checking for device automatically
if torch.cuda.is_available():
    device = "cuda"
    print("CUDA is available. Using GPU.")
else:
    device = "cpu"



#Augment and Show images
num_images = 100
plt.figure(figsize=(20,20))                                                                                                         #window display size of images
          
img_augment = v2.Compose([
        v2.ToImage(),                                  #converts to a torch tensor image object
        v2.ToDtype(torch.float32, scale=True),                                                  
        v2.Resize((224,224)),                          #resizes the image to 224 x 224
        v2.RandomHorizontalFlip(p=0.3),
        v2.RandomVerticalFlip(p=0.4),
        v2.ColorJitter(brightness=0.15,contrast=0.15),
        v2.RandomApply([
            v2.RandomRotation(degrees=50),
            v2.RandomResizedCrop(224, scale=(0.85,1.0)), 
        ], p=0.5),
    ])                     
                                                                                                                                    
for i in range(num_images):
    img_path = 'getImages2/' + df.iloc[i]['img_path']                                                                                    
    name = df.iloc[i]['scientific_name']

    img = PIL.Image.open(img_path)                                                                                                  

    use_transforms = img_augment(img)

    plt.subplot(10, 10, i+1)                                                                                    
    plt.imshow(v2.ToPILImage()(use_transforms))
    plt.title(name[:15], fontsize=6)
    plt.axis("off")

plt.tight_layout(pad=2, h_pad=2.5, w_pad=0.2)
#Use plt.show(block=False) for debugging purposes; this will prevent the graph from popping up
plt.savefig('Transforms')                                                                                          
plt.show(block=False)            #Use plt.show(block=False) for debugging purposes; this will prevent the graph from popping up                                                                  



transforms = v2.Compose([        #Transforms for testing/validation                                                           
        v2.ToImage(),                                                                                           
        v2.ToDtype(torch.float32, scale=True),                                                 
        v2.Resize((224,224)),
    ])


#Define datasets and dataloaders

root = 'imagesOrganizedSplit'
#Create Imagefolders
train_dataset = ImageFolder(os.path.join(root,'train'), transform=img_augment)          #Creates a path to the respective folder
test_dataset = ImageFolder(os.path.join(root,'test'), transform=transforms)             #Only test/val use normal transforms and training uses image augmentations
val_dataset = ImageFolder(os.path.join(root,'val'), transform=transforms)
                                                                                                    
                                                                                                                #Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=64, pin_memory=True, num_workers=16, shuffle=True)      #
test_dataloader = DataLoader(test_dataset, batch_size=16, pin_memory=True, num_workers=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, pin_memory=True, num_workers=16, shuffle=True)

                                                                                                                #Check dataloader outputs 
#Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16,shuffle=True)
 
#Check dataloader outputs 
for images, labels in train_dataloader:
    print(f"\nTrain inputs: {images.size()}")       #Input order: ([batch size, channels, img height, img width])
    print(f"Train outputs: {labels.size()}")        #Output order: ([batch size])
    break

for images, labels in test_dataloader:
    print(f"\nTest inputs: {images.size()}")
    print(f"Test outputs: {labels.size()}")
    break
    
for images, labels in val_dataloader:
    print(f"\nValidation inputs: {images.size()}")
    print(f"Validation outputs: {labels.size()}")
    break



#Define the CNN model class
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1 ,1)
        self.bN1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bN2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bN3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bN4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bN5 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(7*7*512, 500)
        self.fc2 = nn.Linear(500, 26)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, X):                                   #pass convolutions through pooling layers, relu activation and batch norms
        X = self.pool(self.bN1(self.relu(self.conv1(X))))
        X = self.pool(self.bN2(self.relu(self.conv2(X))))
        X = self.pool(self.bN3(self.relu(self.conv3(X))))
        X = self.pool(self.bN4(self.relu(self.conv4(X))))
        X = self.pool(self.bN5(self.relu(self.conv5(X))))
        X = X.flatten(start_dim=1)
        X = self.relu(self.fc1(X))
        output = self.fc2(X)
        return output

model = ConvNet()   

#Check output of the model
for images, label in train_dataloader:
    print(f'\nImage shape: {images.shape}')                  #print dimensions of input image shape
    output_model = model(images)                                                                             
    print(f'Output shape: {output_model.shape}')             #print the output tensor of model shape
    print(output_model[0])                                   #prints image shape for first image in batch
    break

model.to(device) 


#Training, Validation and Testing Loop

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)                                                               
criterion = nn.CrossEntropyLoss().to(device)  

NUM_EPOCHS = 3
NUM_EPOCHS = 1

training_loop_time = time.time()                    #Calculate the time at the beginning of the training loop

#Training Loop
for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()                  #Calculate time at the beginning of each epoch
    model.train()

    train_correct_vals = 0
    train_total_imgs = 0
    train_accuracy = 0
    train_total_loss = 0

    v_correct_vals = 0
    v_total_imgs = 0

    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)


        train_preds = model(images)
        train_loss = criterion(train_preds, labels)

        _, tr_preds = torch.max(train_preds, dim=1)

        train_correct_vals += torch.sum((tr_preds == labels)).item()                                                      
        train_total_imgs += labels.size(0)

        train_total_loss += train_loss.item()

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
    train_accuracy = train_correct_vals / train_total_imgs
    avg_train_loss = train_total_loss / len(train_dataloader)
    epoch_time = time.time() - epoch_start_time

    print(f"Epoch: {epoch+1}/{NUM_EPOCHS} || Training Loss: {train_loss.item():.6f} || Avg Training Loss: {avg_train_loss:.6f} ||" 
          f" Training Accuracy: {train_accuracy:.6f} || Runtime: {(epoch_time/60):.2f} mins")
    model.eval()

#Validation loop
    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)

            val_preds = model(images)
            val_loss = criterion(val_preds, labels)

            __, v_preds = torch.max(val_preds, dim=1)
                    
            v_correct_vals += torch.sum((v_preds == labels)).item()                                                    
            v_total_imgs += labels.size(0)        


print("\nTesting Phase")

with torch.no_grad():
    test_correct_vals = 0
    test_total_imgs = 0

    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)

        test_preds = model(images)
        test_loss = criterion(test_preds, labels)

        __, tt_preds = torch.max(test_preds, dim=1)

        test_correct_vals += torch.sum((tt_preds == labels)).item()
        test_total_imgs += labels.size(0)


    test_accuracy = test_correct_vals / test_total_imgs
    print(f"Test Loss: {test_loss.item()} || Testing Accuracy: {test_accuracy:.6f}")

print(f"Total time: {((time.time() - training_loop_time)/60):.2f}")

torch.save(model.state_dict(), 'save/to/path/CMPM17_FINAL_SAVE.pth')

torch.save(model.state_dict(), "CMPM17_final_save.pt")