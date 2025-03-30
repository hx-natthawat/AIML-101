# Machine Learning 101 (ML101)

A comprehensive collection of Machine Learning laboratories and practical exercises, designed to provide hands-on experience with fundamental ML concepts and techniques.

## Course Structure

### Lab 0: Python for Machine Learning

- Introduction to essential Python libraries for ML:
  - NumPy for numerical computations
  - Pandas for data manipulation and analysis
  - Matplotlib for data visualization
- Hands-on practice with California Housing dataset
- Topics covered:
  - Basic data manipulation with Pandas
  - Data visualization techniques
  - Statistical analysis and data exploration

### Lab 1: Linear Regression

- Implementation of simple linear regression
- Case study: Predicting Fuel Costs Based on Passenger Count
- Topics covered:
  - Data loading and exploration
  - Data visualization using Matplotlib and Seaborn
  - Model building using scikit-learn
  - Model evaluation using MSE and R² metrics

### Lab 2: Multiple Linear Regression

- Advanced regression with California Housing dataset
- Predicting house prices using multiple features
- Topics covered:
  - Data preprocessing and feature scaling
  - Multiple feature analysis
  - Model training and validation
  - Performance evaluation and interpretation

### Lab 3: Classification with Logistic Regression

- Binary classification using Logistic Regression
- Case study: Fruit Classification based on Weight and Size
- Topics covered:
  - Binary classification concepts
  - Feature visualization
  - Model training and evaluation
  - Decision boundary visualization

### Lab 4: Neural Networks

- Implementation of Multi-Layer Perceptron (MLP) using scikit-learn
- Case study: Binary Classification for Student Admission
- Topics covered:
  - Neural network architecture and components
    - Input, hidden, and output layers
    - Activation functions (ReLU)
    - Optimizers (Adam)
  - Data preprocessing and feature engineering
  - Advanced visualization techniques
    - Feature relationship analysis
    - Decision boundary visualization
    - Interactive plotting with Plotly
  - Model training and hyperparameter tuning
  - Performance evaluation and metrics
    - Model accuracy and validation
    - Overfitting prevention

### Lab 5: Neural Networks for Image Classification

- Implementation of neural networks for image classification using different libraries
- Case study: MNIST Handwritten Digit Classification
- Topics covered:
  - Comparative implementations using scikit-learn, Keras/TensorFlow, and PyTorch
  - Neural network architectures for image data
    - MLPClassifier with scikit-learn
    - Sequential model with Keras
    - Custom nn.Module with PyTorch
  - Data preprocessing for image classification
  - Model training and evaluation
  - Performance comparison between different implementations
  - Hardware acceleration options (CPU, GPU, TPU)
  - Framework comparison and selection criteria

### Lab 6: Natural Language Processing (NLP)

- Text processing techniques for English and Thai languages
- Text vectorization and classification techniques
- Topics covered:
  - Text normalization and cleaning
  - Tokenization (sentence and word level)
  - Stopword removal
  - Part-of-speech tagging
  - Stemming for English text
  - Word segmentation for Thai text
  - Handling language-specific challenges
  - TF-IDF vectorization for document representation
  - Text classification using machine learning algorithms
  - Model evaluation and performance comparison

## Project Structure

```
ML101/
├── lab/
│   ├── lab0-intro/
│   │   ├── 101.ipynb        # Python basics and data analysis
│   │   └── 101-2.ipynb     # Advanced Python concepts
│   ├── lab1/
│   │   └── lab1_Regression.ipynb    # Simple linear regression
│   ├── lab2/
│   │   └── lab2_MultipleRg.ipynb    # Multiple regression
│   ├── lab3/
│   │   └── lab3_Classification_Logistic_Fruits.ipynb
│   └── content/            # Additional resources
├── ml/
│   ├── gradient-descent/   # Interactive visualization tool
│   ├── neural-networks/    # Neural network implementations
│   │   └── 001/           # Student admission prediction with MLP
│   ├── computer-visions/   # Computer vision implementations
│   │   └── 001/           # Image classification with neural networks
│   │       ├── NNwithSKlearn.ipynb  # Neural networks with scikit-learn
│   │       ├── Lab7KerasNN.ipynb    # Neural networks with Keras
│   │       ├── pytorch_mnist.py     # Neural networks with PyTorch
│   │       ├── comparison.md        # Comparison between scikit-learn and Keras
│   │       ├── keras_vs_pytorch.md  # Comparison between Keras and PyTorch
│   │       └── README.md            # Detailed documentation
│   └── nlp/               # Natural Language Processing
│       ├── 001/           # Text processing basics
│       │   ├── 01-textprocessing.ipynb  # English text processing with NLTK
│       │   ├── 02-pythainlp.ipynb       # Thai text processing with PyThaiNLP
│       │   ├── 03-tfidf.ipynb           # TF-IDF implementation
│       │   └── 04-movie-review-classificatiob.ipynb  # Text classification
│       └── README.md       # NLP documentation
└── README.md
```

## Gradient Descent Visualization Tool

An interactive web application for visualizing gradient descent optimization in linear regression. Features include:

- Real-time parameter adjustment
- 2D and 3D visualizations of the cost function
- Interactive learning rate optimization
- Cost history tracking

## Dependencies

Required packages:

```
numpy>=1.19.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
streamlit>=1.0.0
plotly>=5.3.0
tensorflow>=2.0.0  # For Keras neural network implementations
keras>=2.3.0       # For advanced neural network models
torch>=1.7.0       # For PyTorch neural network implementations
torchvision>=0.8.0 # For computer vision utilities in PyTorch
```

## Setup and Installation

1. Clone the repository:

```bash
git clone [repository-url]
cd ML101
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. For interactive notebooks:

```bash
jupyter notebook
```

4. For gradient descent visualization:

```bash
streamlit run ml/gradient-descent/gradient_descent_app.py
```

## Prerequisites

- Basic Python programming knowledge
- Understanding of fundamental mathematical concepts
- Jupyter Notebook environment
- Python 3.7 or higher

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is open-source and available under the MIT License.
