'''
Plant Family Identifier Model
Andrea Okolo and Griffin Svec-Burdick
CMPM17 Winter 2026 Final Project
'''

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
import wandb
from sklearn.metrics import f1_score

#For confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Set epochs
NUM_EPOCHS = 1

df = pd.read_csv("DownloadedImageData_NewPaths.csv")             #load data into dataframe

if __name__ == '__main__': #skip these for demo
    run = wandb.init(project="Final Plant Family Model", name="Test Loss and Test Accuracy Graphs")

    #Checking for device automatically
    if torch.cuda.is_available():
        device = "cuda"
        print("CUDA is available. Using GPU.")
    else:
        device = "cpu"



#Augment and Show images
num_images = 100
plt.figure(figsize=(20,20))                 #window display size of images
          
img_augment = v2.Compose([                             #Transforms for training data
        v2.ToImage(),                                  #v2.ToImage() and v2.ToDtype(torch.float32, scale=True) convert to a torch tensor image object
        v2.ToDtype(torch.float32, scale=True),                                                  
        v2.Resize((224,224)),                               #resizes the image to 224 x 224
        v2.RandomHorizontalFlip(p=0.3),                     #30% probability of flipping the image on its x-axis
        v2.RandomVerticalFlip(p=0.4),                       #40% probability of flippng the image on its y-axis
        v2.ColorJitter(brightness=0.15,contrast=0.15),      #Slightly shift the color contrast and brightness to account for variations in photos such as lighting 
        v2.RandomApply([                                    #will apply the following features at the same time
            v2.RandomRotation(degrees=50),                  #will randonmly rotate the image by 50 degrees
            v2.RandomResizedCrop(224, scale=(0.85,1.0)),    #RandomResizedCrop order:(size, )
        ], p=0.5),
    ])                     
                                                                                                                                    
for i in range(num_images):
    img_path = 'getImages2/' + df.iloc[i]['img_path']       #Creates a path to the image folder to display images                                                                                   
    name = df.iloc[i]['scientific_name']                    #locates the column scientific name and retrieves the respective name for the image

    img = PIL.Image.open(img_path)                          #allows the image to be opened                                                                                            

    use_transforms = img_augment(img)

    plt.subplot(10, 10, i+1)                                #plots the images in a 10x10 matrix                                                                           
    plt.imshow(v2.ToPILImage()(use_transforms))             
    plt.title(name[:15], fontsize=6)
    plt.axis("off")

plt.tight_layout(pad=2, h_pad=2.5, w_pad=0.2)               #adds extra spacing between titles and images
plt.savefig('Transforms')        #                                                                                    
plt.show(block=False)            #Use plt.show(block=False) for debugging purposes; this will prevent the graph from popping up                                                                  



transforms = v2.Compose([        #Transforms for testing/validation                                                           
        v2.ToImage(),                                                                                           
        v2.ToDtype(torch.float32, scale=True),                                                 
        v2.Resize((224,224)),
    ])


#Define datasets and dataloaders

root = 'imagesOrganizedSplit'
#Create Imagefolders
train_dataset = ImageFolder(os.path.join(root,'train'), transform=img_augment)       #Creates a path to the respective folder
test_dataset = ImageFolder(os.path.join(root,'test'), transform=transforms)          #Only test/val use normal transforms and training uses image augmentations
val_dataset = ImageFolder(os.path.join(root,'val'), transform=transforms)


#Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=64, pin_memory=True, num_workers=16, shuffle=True)  #Use pin_memory=True & num_workers=16 for helping speed up GPU processes    
test_dataloader = DataLoader(test_dataset, batch_size=16, pin_memory=True, num_workers=16,  shuffle=True)    #Exclude pin_memory=True & num_workers=16 when running on local device otherwise code will not run
val_dataloader = DataLoader(val_dataset, batch_size=16, pin_memory=True, num_workers=16, shuffle=True)       #num_workers helps with memory usage; num_workers operates by splitting the data into multiple subprocesses that run at the same time and speed up data computations. Helpful for faster calculations and large datasets
                                                                                                            #using pin_memory=True allows for tensors to be copied to CUDA before returning an output. Also speeds up computational processes
                                                                                                            #Copy and paste pin_memory=True, num_workers=16, when debugging between cpu and gpu

 

if __name__ == '__main__': #Check dataloader outputs (not if importing this as a module)
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
        super().__init__()                          #nn.Conv2d order: ([in_channels(RGB), out channels, kernel size, stride, padding])
        self.conv1 = nn.Conv2d(3, 32, 3, 1 ,1)      #apply 32 3x3 filters to the image; this doubles in size through each convolution layer
        self.bN1 = nn.BatchNorm2d(32)               #BatchNorm2D adjusts values to have a std of 1 and mean of 0. Helps the model train faster and have more consistent results
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bN2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bN3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bN4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bN5 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2,2)            #Scales down the matrix into a 2x2 block and computes the maximum; Helps the model analyze important features of images
        self.fc1 = nn.Linear(7*7*512, 500)       #Images will pass through 5 layers and will be scaled down by the pooling layers. multiplied by the final number of filters applied to the images. 500 is the number of neurons calcualted.
        self.fc2 = nn.Linear(500, 25)            #takes the 500 neurons created and maps them to the 25 plant classes
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()              #merge the dimensions into a single tensor

    def forward(self, X):                        #Pass convolutions through pooling layers, relu activation, and batch norms
        X = self.pool(self.bN1(self.relu(self.conv1(X))))
        X = self.pool(self.bN2(self.relu(self.conv2(X))))
        X = self.pool(self.bN3(self.relu(self.conv3(X))))
        X = self.pool(self.bN4(self.relu(self.conv4(X))))
        X = self.pool(self.bN5(self.relu(self.conv5(X))))
        X = X.flatten(start_dim=1)               #
        X = self.relu(self.fc1(X))
        output = self.fc2(X)
        return output

if __name__ == '__main__': # Prevents the model from rerunning when importing to the demo file

    model = ConvNet()                               

    #Check output of the model
    for images, label in train_dataloader:
        print(f'\nImage shape: {images.shape}')                  #print dimensions of input image shape
        output_model = model(images)                                                                             
        print(f'Output shape: {output_model.shape}')                                                                #print the output tensor of model shape
        # print(output_model[0])                                                                                      #prints image shape for first image in batch
        break

    model.to(device) 


    #Training, Validation and Testing Loop

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)                                                                  
    criterion = nn.CrossEntropyLoss().to(device)  


    training_loop_time = time.time()            #Calculate the time at the beginning of the training loop

    #Training Loop
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()          #Calculate time at the beginning of each epoch
        model.train()

        train_correct_vals = 0
        train_total_imgs = 0
        train_accuracy = 0
        train_total_loss = 0

        v_correct_vals = 0
        v_total_imgs = 0

        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)   #moves the images and labels to the GPU

            train_preds = model(images)                     #calculate model predictions
            train_loss = criterion(train_preds, labels)     #compare predictions to the actual values and calculate using the previously defined loss function

            _, tr_preds = torch.max(train_preds, dim=1)     #_ ignores the inital value dummy variable; tr_preds will predict the highest/most accurate class

            train_correct_vals += torch.sum((tr_preds == labels)).item()       #Compare predictions to labels. If predictions are correct add to a total value                                      
            train_total_imgs += labels.size(0)                                 #keep track of the images procressed

            train_total_loss += train_loss.item()           #add up the total loss value

            optimizer.zero_grad()       #reset slope calculations
            train_loss.backward()       #calculates slopes  
            optimizer.step()            #updates weights
            
        train_accuracy = train_correct_vals / train_total_imgs          #calcualte train accuracy by dividing total correct values over total images processed
        avg_train_loss = train_total_loss / len(train_dataloader)       #caluclate average training loss by dividing total loss value over the total items in the train_dataloaders
        
        epoch_time = time.time() - epoch_start_time             #Calculate the time at the end of the epoch

        print(f"Epoch: {epoch+1:03d}/{NUM_EPOCHS:03d} || Training Loss: {train_loss.item():.6f} || Avg Training Loss: {avg_train_loss:.6f} ||" 
            f" Training Accuracy: {train_accuracy:.6f} || Runtime: {(epoch_time/60):.2f} mins")
        model.eval()

    #Validation loop
        with torch.no_grad():                                               #disables background gradient calculations; Using in validation loop helps speed up the model output
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device)       #moves the images and labels to the GPU

                val_preds = model(images)
                val_loss = criterion(val_preds, labels)

                __, v_preds = torch.max(val_preds, dim=1)
                        
                v_correct_vals += torch.sum((v_preds == labels)).item()                                                    
                v_total_imgs += labels.size(0)        


    print("\nTesting Phase")

    with torch.no_grad():
        test_correct_vals = 0
        test_total_imgs = 0

        labels_list = []                       #labels and preds for both confusion matrix and f1 score
        preds_list = []                        

        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)           #moves the images and labels to the GPU

            test_preds = model(images)
            test_loss = criterion(test_preds, labels)

            __, tt_preds = torch.max(test_preds, dim=1)

            test_correct_vals += torch.sum((tt_preds == labels)).item()
            test_total_imgs += labels.size(0)

            run.log({"Test Loss": test_loss})


            preds_list.extend(tt_preds.cpu()) # Move to CPU and add to list of predictions
            labels_list.extend(labels.cpu()) # Move to CPU and add to list of labels

        test_accuracy = test_correct_vals / test_total_imgs
        print(f"Test Loss: {test_loss.item()} || Testing Accuracy: {test_accuracy:.6f}")

        for epoch in range(NUM_EPOCHS):
            run.log({"Test Accuracy": test_accuracy})

        label_names = test_dataset.classes
        
        cm = confusion_matrix(labels_list, preds_list)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
        disp.plot()
        plt.xticks(rotation = 'vertical')
        plt.tight_layout() # make tick labels fit
        plt.savefig(f'confusion_matrix/confusion_matrix_{NUM_EPOCHS:03d}epochs.png') #this will overwrite previous

        f1_macro =f1_score(labels_list, preds_list, average="macro")      #compute f1 score
        run.log({"Test f1 score":f1_macro})                                                 



    print(f"Total time: {((time.time() - training_loop_time)/60):.2f}")             #print total time for the whole training loop to process

    run.log({"train loss": train_loss, "test loss": test_loss, "train accuracy": train_accuracy})

    # Save the model based on epoch number and current time
    torch.save(model.state_dict(), f"saved_models/final_save_{NUM_EPOCHS:03d}_epochs_{int(time.time())}.pt")             #Saves the model to a file called final_save.pt


