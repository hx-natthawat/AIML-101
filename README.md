# Gradient Descent Visualization

This project provides an interactive visualization of the gradient descent algorithm using Streamlit. It helps in understanding how different learning rates and number of iterations affect the convergence of the algorithm.

## Features

- Interactive parameter adjustment (learning rate and iterations)
- Real-time visualization of gradient descent process
- Automatic global optima finding
- Detailed iteration history with parameter tracking
- Performance comparison table for different learning rates

## Setup

1. Install the required packages:
```bash
pip install streamlit numpy pandas matplotlib
```

2. Run the application:
```bash
streamlit run gradient_descent_app.py
```

## Project Structure

- `gradient-descent/`
  - `gradient-descent-101.ipynb`: Original Jupyter notebook implementation
  - `gradient_descent_app.py`: Streamlit application for interactive visualization

## Usage

1. Use the sidebar to adjust learning rate and number of iterations
2. Choose between manual optimization or automatic global optima finding
3. Watch the animation to see how the algorithm converges
4. View detailed metrics and performance in the simulation results table
