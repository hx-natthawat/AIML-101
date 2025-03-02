# Keras vs PyTorch: Neural Network Implementation Comparison

This document provides a side-by-side comparison of neural network implementations for MNIST image classification using Keras (with TensorFlow backend) and PyTorch.

## Overview

Both implementations accomplish the same task: classifying handwritten digits from the MNIST dataset using a neural network with similar architecture. The key differences lie in the syntax, API design philosophy, and execution model.

## Implementation Comparison

### 1. Library Imports

**Keras:**
```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf
```

**PyTorch:**
```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
```

### 2. Hardware Acceleration Check

**Keras:**
```python
tf.test.gpu_device_name()
```

**PyTorch:**
```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

### 3. Data Loading

**Keras:**
```python
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
```

**PyTorch:**
```python
transform = transforms.Compose([
    transforms.ToTensor(),  # Scales to [0,1] and converts to tensor
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

# Create data loaders
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
```

### 4. Label Encoding

**Keras:**
```python
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
```

**PyTorch:**
```python
# One-hot encoding is handled by the loss function (CrossEntropyLoss)
# For demonstration:
def to_one_hot(labels, num_classes):
    return F.one_hot(labels, num_classes).float()
```

### 5. Model Definition

**Keras:**
```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(20, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()
```

**PyTorch:**
```python
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

model = NeuralNetwork().to(device)

# Print model summary
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Model Architecture:")
print(model)
print(f"Total trainable parameters: {count_parameters(model)}")
```

### 6. Model Compilation

**Keras:**
```python
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
```

**PyTorch:**
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
```

### 7. Model Training

**Keras:**
```python
batch_size = 100
epochs = 10

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
```

**PyTorch:**
```python
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
        # ... validation code ...
        
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
```

### 8. Model Evaluation

**Keras:**
```python
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
```

**PyTorch:**
```python
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
```

## Key Differences

1. **API Philosophy**:
   - **Keras**: High-level, user-friendly API with a focus on simplicity and rapid prototyping
   - **PyTorch**: More explicit, lower-level API with a focus on flexibility and transparency

2. **Execution Model**:
   - **Keras**: Uses a static computational graph (TensorFlow 2.x uses eager execution by default, but still has a static graph under the hood)
   - **PyTorch**: Uses a dynamic computational graph, allowing for more flexibility in model design

3. **Data Handling**:
   - **Keras**: Works with NumPy arrays directly
   - **PyTorch**: Uses its own Tensor class and DataLoader for efficient batch processing

4. **Model Definition**:
   - **Keras**: Sequential API for simple stacking of layers
   - **PyTorch**: Object-oriented approach with custom classes inheriting from nn.Module

5. **Training Loop**:
   - **Keras**: Built-in fit() method handles the training loop
   - **PyTorch**: Requires manual implementation of the training loop, providing more control

6. **Device Management**:
   - **Keras**: Automatically uses available GPU
   - **PyTorch**: Requires explicit device management (moving model and data to GPU)

## When to Choose Each Framework

### Choose Keras When:
- You need rapid prototyping and development
- You prefer a high-level, user-friendly API
- You want built-in training loops and metrics
- You're new to deep learning and want a gentler learning curve

### Choose PyTorch When:
- You need more flexibility in model design
- You're implementing custom or research-oriented models
- You want more control over the training process
- You prefer dynamic computation graphs for debugging
- You're working on research projects that require fine-grained control

## Conclusion

Both Keras and PyTorch are powerful frameworks for implementing neural networks. Keras offers simplicity and rapid development, while PyTorch provides flexibility and control. The choice between them depends on your specific needs, preferences, and the requirements of your project.

For the MNIST classification task, both implementations achieve similar performance, but the code structure and development approach differ significantly.
