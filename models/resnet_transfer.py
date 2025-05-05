import torch.nn as nn
from torchvision import models as tv_models

class ResNetTransfer(nn.Module):
    def __init__(self, num_classes=2, freeze_base=True):
        super(ResNetTransfer, self).__init__()
        
        # Load pre-trained ResNet18
        self.model = tv_models.resnet18(pretrained=True)

        # Optionally freeze all convolutional layers
        if freeze_base:
            for param in self.model.parameters():
                param.requires_grad = False

        # Replace the final fully connected layer to match binary classification
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)  # Outputs logits for 2 classes
