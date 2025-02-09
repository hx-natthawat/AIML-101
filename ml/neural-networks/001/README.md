# Neural Network Implementation - Lab 001

This lab demonstrates the implementation of a basic neural network for binary classification using scikit-learn's MLPClassifier.

## Overview

The project implements a neural network to classify data based on two subject scores (SubjectA and SubjectB). The implementation includes:

- Data loading and preprocessing
- Model implementation using MLPClassifier
- Data visualization
- Model training and evaluation

## Dataset

The dataset used in this project is from:
https://raw.githubusercontent.com/toche7/DataSets/main/admit.csv

It contains three columns:
- SubjectA: First subject score
- SubjectB: Second subject score
- Label: Binary classification target (0 or 1)

## Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Plotly

## Implementation Details

The neural network is implemented using scikit-learn's MLPClassifier with the following architecture:
- Input layer: 2 features (SubjectA and SubjectB)
- Hidden layers: (2, 50) neurons
- Output layer: Binary classification (0 or 1)

## Usage

Open and run the Jupyter notebook `NeuralNetwork.ipynb` to:
1. Load and visualize the data
2. Train the neural network model
3. Evaluate the model's performance
4. Visualize the decision boundaries
