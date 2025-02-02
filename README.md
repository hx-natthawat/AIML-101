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
│   └── gradient-descent/   # Interactive visualization tool
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
