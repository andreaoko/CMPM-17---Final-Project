import torch
from PIL import Image
from torchvision.transforms import v2

from trainModelFile import MyModel
# REPLACE "trainModelFile" WITH THE NAME OF THE FILE WITH THE MODEL CLASS

# create the model class, and load the weights. make sure "model.pt" matches
# the filename you used when saving the model (should be in the same folder as this file)
model = MyModel()
model.load_state_dict(torch.load("model.pt", weights_only=True))

# set to eval mode (only matters if you are using dropout)
model.eval()

# transforms are only for resizing the image or necessary other commands
# make sure resize pixels here match your model, replace (100,100) with your size!
transforms = v2.Compose([
    v2.ToTensor(),
    v2.Resize((100,100)),
])

# load the file "image.png", change this to your file name
img = Image.open("image.png").convert('RGB')
# apply transformations (resizing) to the image
img = transforms(img)

# print(img.shape) # check image shape is correct, if it isn't, unsqueeze
img = torch.unsqueeze(img, 0)

pred = model(img)
print(pred.item())
# at minimum the output should print a prediction, but if you are doing classification,
# use Softmax to turn the output into percentages 
# (see week 4 day 2 activity document on canvas)
# also, try to convert the raw number output into understandable classes