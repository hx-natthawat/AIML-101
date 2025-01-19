import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generate_data(theta0=4, theta1=3, num_points=100):
    X = 2 * np.random.rand(num_points, 1)
    y = theta0 + theta1 * X + np.random.randn(num_points, 1)
    return X, y

def cal_cost(theta, X, y):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/2*m) * np.sum(np.square(predictions-y))
    return cost

def gradient_descent(X, y, theta, learning_rate=0.01, iterations=100):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, 2))
    prediction_history = []  # Store predictions for each iteration
    
    for it in range(iterations):
        prediction = np.dot(X, theta)
        prediction_history.append(prediction)
        theta = theta - (1/m) * learning_rate * (X.T.dot((prediction - y)))
        theta_history[it,:] = theta.T
        cost_history[it] = cal_cost(theta, X, y)
    
    return theta, cost_history, theta_history, prediction_history

def plot_data_and_line(X, y, theta, title="", iteration_data=None):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(X, y, 'b.')
    plt.plot(X, X * theta[1] + theta[0], 'r')
    plt.xlabel("x", fontsize=14)
    plt.ylabel("y", fontsize=14)
    plt.title(title, fontsize=16)
    
    # Add iteration data as text on the plot if provided
    if iteration_data:
        info_text = (
            f"Iteration: {iteration_data['iteration']}\n"
            f"Learning Rate: {iteration_data['learning_rate']:.4f}\n"
            f"Current θ₀: {iteration_data['theta0']:.4f}\n"
            f"Current θ₁: {iteration_data['theta1']:.4f}\n"
            f"Current Cost: {iteration_data['cost']:.4f}"
        )
        plt.text(0.02, 0.98, info_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return fig

def plot_cost_history(cost_history):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(cost_history)), cost_history)
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Cost", fontsize=14)
    plt.title("Cost History", fontsize=16)
    return plt

def find_global_optima(X, y, X_b, max_attempts=5):
    best_cost = float('inf')
    best_theta = None
    best_cost_history = None
    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2]
    iterations = 500
    
    # Store results for table
    results_data = []
    
    progress_text = st.empty()
    progress_bar = st.progress(0)
    plot_placeholder = st.empty()
    metrics_placeholder = st.empty()
    
    for i, lr in enumerate(learning_rates):
        progress_text.text(f"Trying learning rate: {lr}")
        progress_bar.progress((i + 1) / len(learning_rates))
        
        # Initialize random theta
        theta = np.random.randn(2, 1)
        
        # Run gradient descent
        theta_final, cost_history, theta_history, prediction_history = gradient_descent(X_b, y, theta, lr, iterations)
        
        # Store results
        results_data.append({
            "Learning Rate": lr,
            "Final Cost": float(cost_history[-1]),
            "θ₀": float(theta_final[0][0]),
            "θ₁": float(theta_final[1][0]),
            "Initial Cost": float(cost_history[0]),
            "Cost Reduction (%)": ((float(cost_history[0]) - float(cost_history[-1])) / float(cost_history[0]) * 100)
        })
        
        # Update if this is the best result so far
        if cost_history[-1] < best_cost:
            best_cost = cost_history[-1]
            best_theta = theta_final
            best_cost_history = cost_history
            
            # Show current best result
            fig = plot_data_and_line(X, y, theta_final, f"Best Fit (Learning Rate: {lr})")
            plot_placeholder.pyplot(fig)
            plt.close()
            
            metrics_placeholder.write(f"""
                Current best results:
                - Learning Rate: {lr}
                - Final Cost: {best_cost:.4f}
                - θ₀: {float(best_theta[0][0]):.4f}
                - θ₁: {float(best_theta[1][0]):.4f}
            """)
            
        # Small delay for visualization
        plt.close('all')
    
    # Create and display results table
    results_df = pd.DataFrame(results_data)
    results_df = results_df.round(4)  # Round all numeric columns to 4 decimal places
    
    st.subheader("Simulation Results")
    st.dataframe(
        results_df.style.highlight_min(['Final Cost'], color='lightgreen')
                       .background_gradient(subset=['Cost Reduction (%)'], cmap='RdYlGn')
    )
    
    return best_theta, best_cost_history

# UI Components
st.title("Gradient Descent Visualization")

# Sidebar for parameters
st.sidebar.header("Parameters")
learning_rate = st.sidebar.slider("Learning Rate (α)", 0.001, 0.1, 0.01, 0.001)
n_iterations = st.sidebar.slider("Number of Iterations", 10, 1000, 100, 10)

# Generate data
X, y = generate_data()
X_b = np.c_[np.ones((len(X), 1)), X]
theta = np.random.randn(2, 1)

# Add tabs for different modes
tab1, tab2 = st.tabs(["Manual Optimization", "Find Global Optima"])

with tab1:
    # Run gradient descent with manual parameters
    theta_final, cost_history, theta_history, prediction_history = gradient_descent(X_b, y, theta, learning_rate, n_iterations)

    # Display results
    st.subheader("Results")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Final θ₀:", float(theta_final[0][0]))
    with col2:
        st.write("Final θ₁:", float(theta_final[1][0]))

    st.write("Final Cost:", float(cost_history[-1]))

    # Create two columns for the visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Plot data and regression line
        fig1 = plot_data_and_line(X, y, theta_final, "Data and Fitted Line")
        st.pyplot(fig1)

        # Plot cost history
        fig2 = plot_cost_history(cost_history)
        st.pyplot(fig2)

    # Animate the gradient descent process
    if st.button("Animate Gradient Descent"):
        progress_bar = st.progress(0)
        plot_placeholder = st.empty()
        info_placeholder = st.empty()
        
        # Create a DataFrame to store iteration history
        iteration_history = []
        
        for i in range(len(theta_history)):
            # Update progress bar
            progress_bar.progress((i + 1) / len(theta_history))
            
            # Get current parameters
            current_theta = theta_history[i].reshape(-1, 1)
            current_cost = cost_history[i]
            
            # Store iteration data
            iteration_data = {
                'iteration': i + 1,
                'learning_rate': learning_rate,
                'theta0': float(current_theta[0][0]),
                'theta1': float(current_theta[1][0]),
                'cost': current_cost
            }
            iteration_history.append(iteration_data)
            
            # Create DataFrame for display
            history_df = pd.DataFrame(iteration_history)
            
            # Update visualization
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Plot current state with iteration data
                fig = plot_data_and_line(X, y, current_theta, 
                                       f"Iteration {i+1}", 
                                       iteration_data=iteration_data)
                plot_placeholder.pyplot(fig)
            
            with col2:
                # Show iteration history table
                info_placeholder.dataframe(
                    history_df.style.highlight_max(subset=['cost'], color='red')
                                    .highlight_min(subset=['cost'], color='lightgreen')
                                    .format({
                                        'learning_rate': '{:.4f}',
                                        'theta0': '{:.4f}',
                                        'theta1': '{:.4f}',
                                        'cost': '{:.4f}'
                                    })
                )
            
            plt.close('all')
            
            # Add small delay for animation
            if i < len(theta_history) - 1:
                plt.close('all')

with tab2:
    st.write("""
    Click the button below to automatically find the global optima by trying different learning rates.
    The process will be animated to show the progress.
    """)
    
    if st.button("Find Global Optima"):
        best_theta, best_cost_history = find_global_optima(X, y, X_b)
        
        # Plot final cost history
        st.subheader("Final Cost History")
        fig = plot_cost_history(best_cost_history)
        st.pyplot(fig)
