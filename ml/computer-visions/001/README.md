# Neural Network Implementation for Image Classification

This directory contains two Jupyter notebooks demonstrating different approaches to implementing neural networks for image classification tasks. The notebooks showcase the use of scikit-learn and Keras libraries for building and training neural network models.

## Notebooks

### 1. NNwithSKlearn.ipynb

This notebook demonstrates the implementation of an image classification model using Neural Networks with scikit-learn's MLPClassifier.

**Key Features:**

- Data loading and preprocessing for image data
- Feature extraction and normalization
- Neural network model implementation using scikit-learn's MLPClassifier
- Model training and evaluation
- Visualization of results

### 2. Lab7KerasNN.ipynb

This notebook demonstrates a more advanced implementation of neural networks for image classification using the Keras library with TensorFlow backend.

**Key Features:**

- MNIST dataset loading and preprocessing
- Model building with Keras Sequential API
- Training on different hardware (CPU, GPU, TPU)
- Flexibility to adjust hidden layers
- Performance evaluation and visualization

## Author

Taweesak Samanchuen

## Description

These notebooks provide examples of creating Image Classification models using Neural Networks with different libraries. They demonstrate how to implement neural networks for computer vision tasks using Python's popular machine learning libraries.

## Requirements

```python
python>=3.7
numpy>=1.19.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
tensorflow>=2.0.0
keras>=2.3.0
```

## Implementation Details

### Scikit-learn Neural Network (NNwithSKlearn.ipynb)

**Neural Network Architecture:**

- Input Layer: Features extracted from images
- Hidden Layers: Multiple layers with configurable sizes
- Output Layer: Classification layer for image categories
- Activation: ReLU (Rectified Linear Unit)
- Solver: Adam optimizer

**Model Configuration:**

```python
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(
    hidden_layer_sizes=(100,),  # Single hidden layer with 100 neurons
    activation='relu',          # ReLU activation function
    solver='adam',              # Adam optimizer
    max_iter=1000               # Maximum iterations
)
```

### Keras Neural Network (Lab7KerasNN.ipynb)

**Neural Network Architecture:**

- Input Layer: Flattened 28x28x1 images
- Hidden Layers: Multiple Dense layers with ReLU activation
- Output Layer: 10 neurons with softmax activation for multi-class classification

**Model Configuration:**

```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(20, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
```

## Implementations Comparison

A detailed comparison between the two neural network implementations is available in the [comparison.md](comparison.md) file. This document provides:

- Detailed analysis of both implementations
- Key differences in architecture and approach
- Advantages and limitations of each method
- Performance comparison
- Guidelines on when to use each implementation

## Usage

1. Open the Jupyter notebooks:

   ```bash
   jupyter notebook NNwithSKlearn.ipynb
   # or
   jupyter notebook Lab7KerasNN.ipynb
   ```

2. Follow the notebook sections:
   - Data Loading and Preprocessing
   - Feature Extraction/Data Preparation
   - Model Training
   - Evaluation and Visualization

## Related Resources

- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
- [Keras Documentation](https://keras.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Neural Networks for Image Classification](https://scikit-learn.org/stable/auto_examples/classification/plot_mnist_filters.html)
- [CPU, GPU, TPU Comparison](https://medium.com/super-ai-engineer/gpu-tpu-%E0%B8%84%E0%B8%B7%E0%B8%AD%E0%B8%AD%E0%B8%B0%E0%B9%84%E0%B8%A3-%E0%B8%84%E0%B8%A7%E0%B8%A3%E0%B9%83%E0%B8%8A%E0%B9%89%E0%B8%AD%E0%B8%B0%E0%B9%84%E0%B8%A3%E0%B9%83%E0%B8%99%E0%B8%81%E0%B8%B2%E0%B8%A3-train-model-%E0%B8%81%E0%B8%B1%E0%B8%99%E0%B9%81%E0%B8%99%E0%B9%88-1b652666cbbf)

## License

This project is available as an open educational resource.
