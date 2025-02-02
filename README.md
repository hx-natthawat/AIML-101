# Machine Learning 101 - Gradient Descent Visualization

An interactive web application for visualizing and understanding gradient descent optimization in linear regression. This educational tool helps users explore how gradient descent works through real-time visualizations and parameter adjustments.

## Features

### Data Generation
- Generate synthetic data with customizable parameters:
  - True θ₀ (intercept): Range from -10 to 10
  - True θ₁ (slope): Range from -10 to 10
  - Number of data points: 10 to 500 points
- Visualize the true underlying linear function

### Interactive Visualizations

#### 1. 2D Plot View
- Scatter plot of data points
- Real-time updating regression line
- Dynamic display of:
  - Current parameter values
  - Cost function value
  - Iteration information

#### 2. 3D Surface Plot
- Interactive 3D visualization of the cost function surface
- Axes:
  - X-axis: θ₀ (intercept parameter)
  - Y-axis: θ₁ (slope parameter)
  - Z-axis: Cost value
- Features:
  - Gradient descent path visualization
  - Color-coded surface showing cost landscape
  - Start point (red) and current/end point (green) markers
  - Interactive camera controls

#### 3. Cost History Plot
- Real-time visualization of cost vs. iterations
- Tracks optimization progress
- Helps identify convergence

### Training Controls
- Adjustable learning rate (α)
- Configurable number of iterations
- Two optimization modes:
  1. Manual Mode: Set parameters manually
  2. Global Optima Search: Automatic learning rate optimization

## Technical Details

### Implementation
- Built with Streamlit for interactive web interface
- Uses NumPy for efficient numerical computations
- Plotly for interactive 3D visualizations
- Matplotlib for 2D plots and data visualization
- Pandas for data management

### Algorithm Details
- **Cost Function**: Mean Squared Error (MSE)
  ```python
  cost = (1/2m) * Σ(hθ(x) - y)²
  ```
  where m is the number of data points, hθ(x) is the prediction, and y is the actual value

- **Gradient Descent Update Rule**:
  ```python
  θ = θ - α * (1/m) * X^T * (X*θ - y)
  ```
  where α is the learning rate, X is the input matrix, and y is the target vector

### Global Optima Search
- Automatically tests multiple learning rates: [0.001, 0.01, 0.05, 0.1, 0.2]
- Runs 500 iterations for each learning rate
- Tracks and compares performance metrics:
  - Final cost
  - Parameter values (θ₀, θ₁)
  - Cost reduction percentage
  - Initial vs final cost

## Course Structure

### Lab 0: Python for Machine Learning
- Introduction to essential Python libraries for ML:
  - NumPy for numerical computations
  - Pandas for data manipulation
  - Matplotlib for data visualization
- Practical examples using California Housing dataset
- Basic data analysis and visualization techniques

### Lab 1: Linear Regression
- Implementation of simple linear regression
- Case study: Predicting Fuel Costs Based on Passenger Count
- Topics covered:
  - Data loading and exploration
  - Data visualization using Matplotlib and Seaborn
  - Model building using scikit-learn
  - Model evaluation and interpretation
  - Performance metrics (MSE, R²)

### Lab 2: Multiple Linear Regression
- Advanced regression techniques with multiple variables
- Handling complex datasets
- Feature selection and engineering
- Model validation techniques

### Lab 3: Classification
- Introduction to classification problems
- Case study: Fruit Classification using Logistic Regression
- Topics covered:
  - Binary and multi-class classification
  - Model evaluation metrics for classification
  - Confusion matrix interpretation

## Project Structure
```
ML101/
├── lab/
│   ├── lab0-intro/
│   │   ├── 101.ipynb        # Python basics for ML
│   │   └── 101-2.ipynb     # Advanced Python concepts
│   ├── lab1/
│   │   └── lab1_Regression.ipynb    # Linear regression exercises
│   ├── lab2/
│   │   └── lab2_MultipleRg.ipynb    # Multiple regression analysis
│   ├── lab3/
│   │   └── lab3_Classification_Logistic_Fruits.ipynb    # Classification exercises
│   └── content/            # Additional learning materials
├── ml/
│   └── gradient-descent/
│       └── gradient_descent_app.py  # Main application file
└── README.md
```

## Setup

### Prerequisites
Make sure you have Python 3.7+ installed on your system.

### Installation
1. Clone the repository:
```bash
git clone [repository-url]
cd ML101
```

2. Install required packages:
```bash
pip install streamlit>=1.0.0 numpy>=1.19.0 pandas>=1.3.0 matplotlib>=3.4.0 plotly>=5.3.0
```

3. Run the application:
```bash
streamlit run ml/gradient-descent/gradient_descent_app.py
```

## Usage Guide

1. **Data Generation**:
   - Use the sidebar sliders to adjust true θ₀ and θ₁ values
   - Set the desired number of data points
   - The true function equation will be displayed

2. **Training Configuration**:
   - Adjust the learning rate (smaller values for more stable learning)
   - Set the number of iterations
   - Choose between manual optimization or global optima search

3. **Visualization Options**:
   - Toggle between 2D and 3D views using tabs
   - Observe the animation of parameter evolution
   - Monitor the cost history
   - Track detailed progress in real-time

4. **Best Practices**:
   - Start with a small learning rate (0.01) and increase gradually
   - Use more data points for stable learning
   - Use the 3D visualization to understand the optimization landscape
   - Try the global optima search for automatic learning rate selection

## Contributing
Contributions are welcome! Please feel free to submit issues and pull requests.

## License
This project is open-source and available under the MIT License.
