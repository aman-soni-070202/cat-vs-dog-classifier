import torch
from torchvision import transforms
from PIL import Image
import os
import sys

from models.cnn_model_v1 import CNNModelV1
from models.cnn_model_v2 import CNNModelV2
from models.resnet_transfer import ResNetTransfer

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the same transforms used in training
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load model
model = ResNetTransfer().to(device)
pth_path = 'models/cat_dog_cnn_resnet.pth'
checkpoint = torch.load(pth_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Class labels
class_names = ['cat', 'dog']

# Prediction function
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]
    
    return predicted_class

# Run prediction
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python predict.py path_to_image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        sys.exit(1)

    prediction = predict_image(image_path)
    print(f"✅ Predicted class: {prediction}")
