import torch
from torchvision import transforms
from PIL import Image
import os
import argparse

from models.cnn_model_v1 import CNNModelV1
from models.cnn_model_v2 import CNNModelV2
from models.resnet_transfer import ResNetTransfer

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# Class labels
class_names = ['cat', 'dog']

# Model loader
def load_model(model_type, model_path):
    if model_type == 'v1':
        model = CNNModelV1()
    elif model_type == 'v2':
        model = CNNModelV2()
    elif model_type == 'resnet':
        model = ResNetTransfer()
    else:
        raise ValueError("Invalid model_type. Choose from: v1, v2, resnet")

    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.to(device)
    model.eval()
    return model

# Predict single image
def predict_image(image_path, model):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]

    return predicted_class

# Main entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cat vs Dog Image Predictor")
    parser.add_argument('--img_path', type=str, required=True, help="Path to an image or folder of images")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model (.pth)")
    parser.add_argument('--model_type', type=str, default='resnet', choices=['v1', 'v2', 'resnet'], help="Model type: v1, v2, or resnet")
    
    args = parser.parse_args()

    if not os.path.exists(args.img_path):
        print(f"❌ File or folder not found: {args.img_path}")
        exit(1)

    try:
        model = load_model(args.model_type, args.model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        exit(1)

    if os.path.isdir(args.img_path):
        image_files = [f for f in os.listdir(args.img_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            print(f"⚠️ No valid images found in {args.img_path}")
            exit(0)

        for file in image_files:
            full_path = os.path.join(args.img_path, file)
            prediction = predict_image(full_path, model)
            print(f"✅ Image: {file} → Prediction: {prediction}")

    else:
        prediction = predict_image(args.img_path, model)
        print(f"✅ Image: {os.path.basename(args.img_path)} → Prediction: {prediction}")
