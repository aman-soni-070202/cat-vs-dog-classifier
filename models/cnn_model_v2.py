import torch.nn as nn
import torch.nn.functional as F

class CNNModelV2(nn.Module):
    def __init__(self):
        super(CNNModelV2, self).__init__()
        
        # Adds multiple channels based on many filters - here increasing to 16 in first layer and then to 32 in the second layer
        # Batch norm adjusts the output of a layer to make it easier for the next layer to learn things. 
        # it shifts the mean and standard-deviation of the inputs of the entire batch closer to 0.
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)


        # reduces spatial size - here reducing it by half
        self.pool = nn.MaxPool2d(2, 2)

        # Shuts down or defuses certain number of neurons randomly on each epoch. Here its 30%.
        # Why? - So that over the period of training the model does not start giving more importance to aa specific neuron.
        self.dropout = nn.Dropout(0.3)


        # flattens the 3d image
        # Q1. How is the image 3d?
        # Ans. because its has a a shape of (128x128)(just an example) which gives it 2 dimensions and then it has multiple channels that gives it a depth dimension
        # After 3 poolings, input goes from 128x128 → 64x64 → 32x32 → 16x16
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)  # flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)  # logits
        return x
