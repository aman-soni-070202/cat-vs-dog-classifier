# 🐱🐶 Cat vs Dog Image Classifier

This project is a deep learning-based binary image classifier that distinguishes between images of cats and dogs. The goal was to explore how different convolutional neural network (CNN) architectures perform on this task, including:

- A simple custom CNN with 2 convolutional layers
- A more robust CNN with 3 layers, Batch Normalization, and Dropout
- A transfer learning model using **ResNet18** pretrained on ImageNet

---

## 🧠 Models Overview

### 1. **CNNModelV1 - Basic CNN (2 Convolutional Layers)**
- Designed from scratch using PyTorch.
- Two convolutional layers followed by pooling and fully connected layers.
- Lightweight and easy to train.
- Suitable for learning basics of CNNs.

### 2. **CNNModelV2 - Deep CNN with Regularization**
- Three convolutional layers with:
  - **Batch Normalization** for faster and stable training
  - **Dropout** for regularization and to reduce overfitting
- Better performance on validation set due to improved generalization.

### 3. **Transfer Learning with ResNet18**
- Utilizes a pretrained **ResNet18** model from `torchvision.models`.
- Replaces the final classification layer to output two classes: cat or dog.
- Fine-tuned only the final layer (optionally more if needed).
- Best suited when training data is limited or for quicker convergence with high accuracy.

---

## 🧾 Dataset Structure

Dataset should follow the folder structure compatible with `torchvision.datasets.ImageFolder`:

```
├── data/
│ └── train_set/ # Training images (cats and dogs in subfolders)
│ └── test_set/ # Testing images (cats and dogs in subfolders)
├── models/
│ ├── cnn_model_v1.py # Basic 2-layer CNN
│ ├── cnn_model_v2.py # 3-layer CNN with BatchNorm and Dropout
│ └── resnet_transfer.py # Pretrained ResNet18 with fine-tuned final layer
├── train.py # Training loop
├── test.py # Model evaluation script
├── predict.py # Predicts class of a given image or folder of images
├── utils.py # Utility functions (like loading model/optimizer)
├── requirements.txt
└── README.md
```

---


Images should be resized to `128x128` or similar for consistent input shape.

---

## 🚀 Training

- Optimizer: **Adam**
- Loss Function: **CrossEntropyLoss**
- Scheduler: **StepLR** (step size = 5, gamma = 0.5)
- Device support: Runs on GPU if available
- Saves the model with the best training loss

Training logs show:
- Per-epoch **training loss**
- **Validation accuracy** for evaluation
- Automatic model checkpointing

---

## 📁 Files

- `models/cnn_model_v1.py` — Simple CNN
- `models/cnn_model_v2.py` — Deeper CNN with BatchNorm and Dropout
- `models/resnet_transfer.py` — Transfer learning with ResNet18
- `utils.py` — Helper methods for loading data, saving/loading checkpoints, etc.
- `train.py` — Main training loop for selected model

---

## 📈 Results

| Model           | Params | Train Time | Accuracy (Val)  | Notes                              |
|-----------------|--------|------------|-----------------|------------------------------------|
| CNNModelV1      | Low    | Fast       | ~85-90%         | Good for simple cases              |
| CNNModelV2      | Medium | Moderate   | ~95-96%         | Better generalization              |
| ResNet18 (TL)   | High   | Fast       | ~97%+           | Best performance, fast convergence |

---

🔍 How to Predict

To classify new images using the trained model, use predict.py.

`python predict.py --img_path "./samples/my_image.jpg" --model_path "models/cat_dog_cnn_resnet.pth"`

Arguments:

    --img_path: Path to the image or folder of images

    --model_path: Path to the trained .pth file

    --model_type: Choose from v1, v2, or resnet (default: resnet)

Example:

`python predict.py --img_path "./samples/cat1.jpg" --model_type resnet`

Output:

```✅ Image: cat1.jpg → Prediction: Cat```

---

## 📦 Requirements

`pip install -r requirements.txt`

---

## 📌 Notes

    The models expect images to be resized to 128x128.

    If you’re using the ResNet model, only the final layer is trained. You can optionally unfreeze deeper layers to fine-tune further.

    The model assumes label 0 is cat and label 1 is dog based on folder names in training data.

---

## 🛠 Future Improvements
- Implement **early stopping**
- Add **data augmentation**
- Explore **other pretrained models** (e.g., VGG16, EfficientNet)
- Build a web interface for uploading images and getting predictions

---

## 📸 Example Usage (Coming Soon)
> Upload an image and see whether it’s a 🐱 or 🐶!

---

## 🙌 Acknowledgements
- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- [Kaggle Cats vs Dogs Dataset](https://www.kaggle.com/c/dogs-vs-cats)

---

## 📄 License
MIT License

---

## 🙌 Credits

Created by Aman as a part of learning deep learning fundamentals from scratch.
