'''Example file from Anthony'''

import torch
import torch.nn as nn

# generate random fake data
data_x = torch.randn((100, 3, 100, 100))
data_y = torch.randn((100, 1))

# your model class (which will be imported into the other file)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, 1, 1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.layer1 = nn.Linear(8*50*50, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.flatten(start_dim=1)
        x = self.layer1(x)
        return x

# This if-statement means the code that is indented in it, will only run if you 
# run the file directly, but not when you import it. So, your training/testing loop
# should be indented within this exact if-statement
if __name__ == '__main__':

    model = MyModel()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # short training loop
    for i in range(10):
        pred = model(data_x)
        loss = loss_fn(pred, data_y)

        loss.backward()
        optim.step()
        optim.zero_grad()

    # testing loop would go here as well

    print("finished training, saving model")

    torch.save(model.state_dict(), "model.pt")