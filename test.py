import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.cnn_model_v1 import CNNModelV1
from models.cnn_model_v2 import CNNModelV2
from models.resnet_transfer import ResNetTransfer

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the same transforms used in training
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# Load the testing data
test_dataset = datasets.ImageFolder(root='./data/test_set/test_set', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load model
# model = CNNModelV1().to(device)
# model = CNNModelV2().to(device)
model = ResNetTransfer().to(device)
pth_path = 'models/cat_dog_cnn_resnet_v1.pth'
checkpoint = torch.load(pth_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Evaluation
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
