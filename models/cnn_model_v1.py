import torch.nn as nn
import torch.nn.functional as F

class CNNModelV1(nn.Module):
    def __init__(self):
        super(CNNModelV1, self).__init__()
        # Adds multiple channels based on many filters - here increasing to 16 in first layer and then to 32 in the second layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # reduces spatial size - here reducing it by half
        self.pool = nn.MaxPool2d(2, 2)

        # flattens the 3d image - 
        # Q1. How is the image 3d? - because its has a a shape of (128x128)(just an example) which gives it 2 dimensions and then it has multiple channels that gives it a depth dimension
        self.fc1 = nn.Linear(32 * 32 * 32, 128)  # assuming 128x128 input images
        self.fc2 = nn.Linear(128, 2)  # 2 output classes (cat, dog)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # logits (softmax will be applied in the loss)
