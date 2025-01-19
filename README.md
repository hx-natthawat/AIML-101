# Gradient Descent Visualization

An interactive visualization tool for understanding gradient descent optimization in linear regression. This application helps users explore how gradient descent works by providing real-time visualizations and parameter adjustments.

## Features

### Data Generation
- Custom parameter settings for generating synthetic data:
  - True θ₀ (intercept): Adjustable from -10 to 10
  - True θ₁ (slope): Adjustable from -10 to 10
  - Number of data points: Configurable from 10 to 500
- Visualization of true underlying function

### Interactive Visualizations
1. 2D Visualization:
   - Scatter plot of data points
   - Real-time updating regression line
   - Current parameter values and cost displayed on plot

2. 3D Visualization:
   - Surface plot of the cost function
   - X-axis: θ₀ (intercept parameter)
   - Y-axis: θ₁ (slope parameter)
   - Z-axis: Cost value
   - Gradient descent path visualization
   - Color gradient showing cost landscape
   - Start point (red) and current/end point (green)

3. Cost History:
   - Plot of cost vs. iterations
   - Real-time updates during optimization

### Training Controls
- Adjustable learning rate (α): Fine-tune the step size
- Configurable number of iterations
- Two optimization modes:
  1. Manual Optimization: Set parameters yourself
  2. Global Optima Search: Automatically tries different learning rates

### Performance Tracking
- Real-time iteration history table
- Color-coded performance metrics:
  - Best results highlighted in green
  - Worst results highlighted in red
- Detailed metrics for each iteration:
  - Current parameter values (θ₀, θ₁)
  - Cost value
  - Learning rate
  - Iteration number

## Setup

1. Install required packages:
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
  - `gradient_descent_app.py`: Streamlit application with interactive visualizations

## Usage Guide

1. Data Generation:
   - Use the sidebar sliders to set true θ₀ and θ₁ values
   - Adjust the number of data points
   - The true function equation will be displayed in the sidebar

2. Training Configuration:
   - Set the learning rate (smaller values for more stable but slower learning)
   - Choose the number of iterations
   - Select between manual optimization or automatic global optima search

3. Visualization Options:
   - Switch between 2D and 3D views using tabs
   - Watch the animation to see how parameters evolve
   - Monitor the cost history
   - Track detailed progress in the iteration history table

4. Tips for Best Results:
   - Start with a small learning rate (0.01) and increase gradually
   - Use more data points for more stable learning
   - Watch the 3D visualization to understand how gradient descent navigates the cost surface
   - Use the global optima search to find the best learning rate automatically

## Contributing

Feel free to submit issues and enhancement requests!
