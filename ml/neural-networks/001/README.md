# Neural Network Implementation - Lab 001

This lab demonstrates the implementation of a basic neural network for binary classification using scikit-learn's MLPClassifier. The project focuses on building a model to predict binary outcomes based on two input features.

## Overview

The project implements a neural network to classify data based on two subject scores. Key components include:

- Data loading and exploratory data analysis (EDA)
- Data preprocessing and feature engineering
- Neural network model implementation using MLPClassifier
- Advanced data visualization using Seaborn and Plotly
- Model training, evaluation, and performance analysis

## Dataset

The dataset used in this project is from:
https://raw.githubusercontent.com/toche7/DataSets/main/admit.csv

Features:
- SubjectA: First subject score (numerical, range: 30-80)
- SubjectB: Second subject score (numerical, range: 40-90)
- Label: Binary classification target (0: Not Admitted, 1: Admitted)

## Requirements

```python
python>=3.7
numpy>=1.19.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
plotly>=5.3.0
```

## Implementation Details

### Neural Network Architecture
- Input Layer: 2 neurons (SubjectA and SubjectB features)
- Hidden Layers: Two layers
  - First hidden layer: 2 neurons
  - Second hidden layer: 50 neurons
- Output Layer: 1 neuron (binary classification)

### Model Configuration
```python
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(
    hidden_layer_sizes=(2, 50),  # Two hidden layers
    activation='relu',           # ReLU activation function
    solver='adam',              # Adam optimizer
    max_iter=1000               # Maximum iterations
)
```

### Performance
- Model Accuracy: 89%
- Features advanced visualization techniques:
  - Pair plots for feature relationships
  - Decision boundary visualization
  - Interactive plots using Plotly

## Usage

1. Clone the repository and navigate to the project directory
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook NeuralNetwork.ipynb
   ```
4. Follow the notebook sections:
   - Data Loading and EDA
   - Data Preprocessing
   - Model Training
   - Visualization and Evaluation

## Project Structure
```
001/
├── NeuralNetwork.ipynb     # Main notebook with implementation
├── README.md              # Project documentation
└── requirements.txt       # Project dependencies
```

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
