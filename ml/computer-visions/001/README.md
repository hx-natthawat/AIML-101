# Neural Network with Scikit-learn for Image Classification

This lab demonstrates the implementation of an image classification model using Neural Networks with scikit-learn's MLPClassifier. The notebook provides a practical example of applying neural networks to computer vision tasks.

## Overview

This notebook focuses on building an image classification model using Neural Networks with the scikit-learn library. Key components include:

- Data loading and preprocessing for image data
- Feature extraction and normalization
- Neural network model implementation using MLPClassifier
- Model training and evaluation
- Visualization of results

## Author

Taweesak Samanchuen

## Description

This document provides an example of creating an Image Classification model using Neural Networks with the scikit-learn library. It demonstrates how to implement a neural network for computer vision tasks using Python's popular machine learning library.

## Requirements

```python
python>=3.7
numpy>=1.19.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
```

## Implementation Details

### Neural Network Architecture

- Input Layer: Features extracted from images
- Hidden Layers: Multiple layers with configurable sizes
- Output Layer: Classification layer for image categories
- Activation: ReLU (Rectified Linear Unit)
- Solver: Adam optimizer

### Model Configuration

```python
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(
    hidden_layer_sizes=(100,),  # Single hidden layer with 100 neurons
    activation='relu',          # ReLU activation function
    solver='adam',              # Adam optimizer
    max_iter=1000               # Maximum iterations
)
```

## Usage

1. Open the Jupyter notebook:

   ```bash
   jupyter notebook NNwithSKlearn.ipynb
   ```

2. Follow the notebook sections:
   - Data Loading and Preprocessing
   - Feature Extraction
   - Model Training
   - Evaluation and Visualization

## Related Resources

- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
- [Neural Networks for Image Classification](https://scikit-learn.org/stable/auto_examples/classification/plot_mnist_filters.html)

## License

This project is available as an open educational resource.
