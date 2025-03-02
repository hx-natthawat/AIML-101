#!/usr/bin/env python
# coding: utf-8

# # Lab 7 Neural Network Using PyTorch Library
# 
# Taweesak Samanchuen
# 
# This document demonstrates creating an Image Classification model with Neural Network using PyTorch library
# 1. Can test performance running on CPU and GPU
# 2. Can easily add or reduce the number of Hidden Layers
# 
# [Read more about CPU GPU TPU](https://medium.com/super-ai-engineer/gpu-tpu-%E0%B8%84%E0%B8%B7%E0%B8%AD%E0%B8%AD%E0%B8%B0%E0%B9%84%E0%B8%A3-%E0%B8%84%E0%B8%A7%E0%B8%A3%E0%B9%83%E0%B8%8A%E0%B9%89%E0%B8%AD%E0%B8%B0%E0%B9%84%E0%B8%A3%E0%B9%83%E0%B8%99%E0%B8%81%E0%B8%B2%E0%B8%A3-train-model-%E0%B8%81%E0%B8%B1%E0%B8%99%E0%B9%81%E0%B8%99%E0%B9%88-1b652666cbbf)

# ![Neural Network](https://zitaoshen.rbind.io/project/machine_learning/how-to-build-your-own-neural-net-from-the-scrach/featured.png)
# 
# credit: https://zitaoshen.rbind.io/project/machine_learning/how-to-build-your-own-neural-net-from-the-scrach/

# Import necessary libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define constants
num_classes = 10
input_shape = (1, 28, 28)  # PyTorch uses channels-first format

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load training data
train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True,
    download=True, 
    transform=transform
)

# Load test data
test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False,
    download=True, 
    transform=transform
)

# Display some sample images
def display_samples(dataset, num_samples=8):
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    axes = axes.flatten()
    
    for i in range(num_samples):
        img, label = dataset[i]
        axes[i].imshow(img.squeeze().numpy(), cmap='gist_gray')
        axes[i].set_xlabel(f"label: {label}")
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    plt.tight_layout()
    plt.show()

# Display sample images
display_samples(train_dataset)

# Create data loaders
batch_size = 100
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# Print dataset information
print(f"Training dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# Check the shape of the data
images, labels = next(iter(train_loader))
print(f"Image batch shape: {images.shape}")
print(f"Label batch shape: {labels.shape}")

# Display a sample label
print(f"Original label: {labels[1]}")

# One-hot encode the labels (equivalent to Keras to_categorical)
def to_one_hot(labels, num_classes):
    return F.one_hot(labels, num_classes).float()

# Show one-hot encoded label
one_hot_labels = to_one_hot(labels, num_classes)
print(f"One-hot encoded label: {one_hot_labels[1]}")

# Define the neural network model (equivalent to Keras Sequential model)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 20)
        self.fc3 = nn.Linear(20, 10)
        
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

# Create the model and move it to the device (CPU or GPU)
model = NeuralNetwork().to(device)

# Print model summary (equivalent to model.summary() in Keras)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Model Architecture:")
print(model)
print(f"Total trainable parameters: {count_parameters(model)}")

# Define loss function and optimizer (equivalent to model.compile in Keras)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training function (equivalent to model.fit in Keras)
def train_model(model, train_loader, criterion, optimizer, epochs=10, device=device):
    # For tracking training progress
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # Split training data for validation (10%)
    train_size = int(0.9 * len(train_loader.dataset))
    val_size = len(train_loader.dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_loader.dataset, [train_size, val_size])
    
    # Create new data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Statistics
                val_running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Loss: {epoch_loss:.4f} - "
              f"Accuracy: {epoch_acc:.4f} - "
              f"Val Loss: {val_epoch_loss:.4f} - "
              f"Val Accuracy: {val_epoch_acc:.4f}")
    
    return train_losses, train_accs, val_losses, val_accs

# Train the model
epochs = 10
train_losses, train_accs, val_losses, val_accs = train_model(
    model, train_loader, criterion, optimizer, epochs, device
)

# Evaluate the model (equivalent to model.evaluate in Keras)
def evaluate_model(model, test_loader, criterion, device=device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = correct / total
    
    return test_loss, test_accuracy

# Evaluate on test data
test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
