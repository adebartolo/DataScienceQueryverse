# =============================
# A Python program that trains a deep learning model on skin condition images 
# and classifies user-provided images to identify potential skin conditions (WIP).
# =============================

# =============================
# Install Required Libraries
# =============================

# Core deep learning, model handling, datasets, UI & evaluation libraries
!pip install torch torchvision transformers datasets pillow requests huggingface_hub scikit-learn streamlit

# =============================
# Imports
# =============================

# --- Core Python Utilities ---
import os
import sys
import time
import json
import yaml
from io import BytesIO

# --- PyTorch & Mixed Precision ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

# --- TorchVision: Models, Transforms, Datasets ---
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

# --- Hugging Face Tools ---
from transformers import pipeline
from datasets import load_dataset
from huggingface_hub import hf_hub_download

# --- Image Handling & Web Requests ---
from PIL import Image
import requests

# --- Evaluation & Visualization ---
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Streamlit UI ---
import streamlit as st

# =============================
# Datasets
# =============================

# load the separate splits if the dataset has train/validation/test splits
data = "ahmed-ai/skin-lesions-classification-dataset"
train_dataset = load_dataset(data, split="train")
valid_dataset = load_dataset(data, split="validation")
test_dataset  = load_dataset(data, split="test")

# =============================
# Model
# =============================

# PyTorch Dataset Wrapper
class SkinDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        label = self.dataset[idx]['label']
        if self.transform:
            image = self.transform(image)
        return image, label

# Image Transform Pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Wrap Datasets (replace with your loaded train_dataset, valid_dataset, test_dataset)
train_data = SkinDataset(train_dataset, transform)
valid_data = SkinDataset(valid_dataset, transform)
test_data  = SkinDataset(test_dataset, transform)

# Dataloader settings
batch_size = 32 
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True, persistent_workers=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, num_workers=2, pin_memory=True, persistent_workers=True)
test_loader  = DataLoader(test_data, batch_size=batch_size, num_workers=2, pin_memory=True, persistent_workers=True)

# Model Setup (ResNet18)
num_classes = len(train_dataset.features['label'].names)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_type = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Loss, Optimizer, Scaler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scaler = GradScaler(enabled=torch.cuda.is_available())

# Checkpoint config
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Resume training from checkpoint (optional)
resume_epoch = 0
checkpoint_path = os.path.join(checkpoint_dir, f"resnet18_epoch{resume_epoch}.pth")
if resume_epoch > 0 and os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Resumed model from {checkpoint_path}")

# Epoch loop
epoch_loop = 10

print("Device:", device)
print("CUDA available:", torch.cuda.is_available())

print("Starting training...")
for epoch in range(resume_epoch, epoch_loop):
    start_time = time.time()

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(train_loader)

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(device_type=device_type, enabled=torch.cuda.is_available()):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % 10 == 0:
            batch_loss = running_loss / (batch_idx + 1)
            print(f"Epoch [{epoch+1}/{epoch_loop}], Batch [{batch_idx+1}/{total_batches}] | Loss: {batch_loss:.4f}")

    end_time = time.time()
    epoch_time_minutes = (end_time - start_time) / 60.0
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}% | Time: {epoch_time_minutes:.2f} minutes")

    # Save checkpoint
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"resnet18_epoch{epoch+1}.pth"))
    print(f"Checkpoint saved for epoch {epoch+1}")

    # Free up memory (optional)
    torch.cuda.empty_cache()

print("Training complete.")
# Exit the program
try:
    sys.exit()
except SystemExit as e:
    print(f"Exit program {e}")

# =============================
# Evaluation
# =============================

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Test Accuracy: {100 * correct / total:.2f}%')

# Set model to evaluation mode
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:  # you can also use valid_loader
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Convert to numpy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Generate Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

# Classification Report
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=train_dataset.features['label'].names))

# Plot Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=train_dataset.features['label'].names,
            yticklabels=train_dataset.features['label'].names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# =============================
# Next Steps
# =============================

# Predict Function for New Images
def predict_image(image_path):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(1).item()

    label_names = train_dataset.features['label'].names
    return label_names[predicted_class]

# Example Usage:
path="/content/bcc images.jpeg" #"/content/reddit bcc.png"
result = predict_image(path)
print("Predicted:", result)

# Save for future use
torch.save(model.state_dict(), "final_model.pth")
