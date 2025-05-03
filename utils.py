import os
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from models.cnn_model import CNNModel
from models.cnn_model_v2 import CNNModelV2


class Utils:
    def __init__(self):
        pass

    @staticmethod
    def training_data_loader(dir_path: str):
        # Image transformations
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor()
        ])

        val_test_transforms = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        # Load full dataset
        full_dataset = datasets.ImageFolder(dir_path, transform=train_transforms)

        # Split 80% train, 20% validation
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size

        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        # Override the transform for validation set (no augmentation)
        val_dataset.dataset.transform = val_test_transforms

        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        return train_loader, val_loader

    @staticmethod
    def get_model_optimizer_best_loss(device: torch.device, pth_path: str, learning_rate: float):
        # model = CNNModel()
        model = CNNModelV2()

        checkpoint = None
        if os.path.exists(pth_path):
            checkpoint = torch.load(pth_path)
            print("✅ Loaded checkpoint from:", pth_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        best_loss = float('inf')

        if checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'best_loss' in checkpoint:
                best_loss = checkpoint['best_loss']

        return model, optimizer, best_loss