import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def generate_data(theta0=4, theta1=3, num_points=100):
    X = 2 * np.random.rand(num_points, 1)
    y = theta0 + theta1 * X + np.random.randn(num_points, 1)
    return X, y

def cal_cost(theta, X, y):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/2*m) * np.sum(np.square(predictions-y))
    return cost

def plot_cost_surface(X, y, theta_history=None):
    # Create a grid of theta0 and theta1 values
    theta0_range = np.linspace(-10, 10, 100)
    theta1_range = np.linspace(-10, 10, 100)
    theta0_mesh, theta1_mesh = np.meshgrid(theta0_range, theta1_range)
    
    # Calculate cost for each combination of theta0 and theta1
    cost_mesh = np.zeros_like(theta0_mesh)
    for i in range(len(theta0_range)):
        for j in range(len(theta1_range)):
            theta = np.array([[theta0_mesh[i,j]], [theta1_mesh[i,j]]])
            cost_mesh[i,j] = cal_cost(theta, X, y)
    
    # Create 3D surface plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surface = ax.plot_surface(theta0_mesh, theta1_mesh, cost_mesh, 
                            cmap='viridis', alpha=0.8)
    
    # Add a color bar
    plt.colorbar(surface)
    
    # If theta history is provided, plot the optimization path
    if theta_history is not None:
        theta0_history = theta_history[:,0]
        theta1_history = theta_history[:,1]
        cost_history = [cal_cost(np.array([[t0], [t1]]), X, y) 
                       for t0, t1 in zip(theta0_history, theta1_history)]
        
        # Plot the path taken by gradient descent
        ax.plot(theta0_history, theta1_history, cost_history, 
                'r.-', linewidth=2, label='Gradient descent path')
        
        # Plot start and end points
        ax.scatter(theta0_history[0], theta1_history[0], cost_history[0], 
                  color='red', s=100, label='Start')
        ax.scatter(theta0_history[-1], theta1_history[-1], cost_history[-1], 
                  color='green', s=100, label='End')
    
    ax.set_xlabel('θ₀')
    ax.set_ylabel('θ₁')
    ax.set_zlabel('Cost')
    ax.set_title('Cost Function Surface')
    ax.legend()
    
    return fig

def gradient_descent(X, y, theta, learning_rate=0.01, iterations=100):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, 2))
    prediction_history = []
    
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

# Data Generation Parameters
st.sidebar.subheader("Data Generation")
theta0_true = st.sidebar.slider("True θ₀", -10.0, 10.0, 4.0, 0.1, 
                              help="True value of θ₀ used to generate the data")
theta1_true = st.sidebar.slider("True θ₁", -10.0, 10.0, 3.0, 0.1,
                              help="True value of θ₁ used to generate the data")
num_points = st.sidebar.slider("Number of Data Points", 10, 500, 100, 10,
                             help="Number of random data points to generate")

# Training Parameters
st.sidebar.subheader("Training")
learning_rate = st.sidebar.slider("Learning Rate (α)", 0.001, 0.1, 0.01, 0.001,
                                help="Step size for gradient descent updates")
n_iterations = st.sidebar.slider("Number of Iterations", 10, 1000, 100, 10,
                               help="Number of gradient descent iterations")

# Generate data with user-specified parameters
X, y = generate_data(theta0=theta0_true, theta1=theta1_true, num_points=num_points)
X_b = np.c_[np.ones((len(X), 1)), X]
theta = np.random.randn(2, 1)

# Add info about true parameters
st.sidebar.markdown("---")
st.sidebar.subheader("True Parameters")
st.sidebar.write(f"True function: y = {theta0_true:.2f} + {theta1_true:.2f}x")

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

    # Create tabs for different visualizations
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["2D View", "3D View", "Cost History"])
    
    with viz_tab1:
        # Plot data and regression line
        fig1 = plot_data_and_line(X, y, theta_final, "Data and Fitted Line")
        st.pyplot(fig1)
    
    with viz_tab2:
        # Plot 3D cost surface with gradient descent path
        fig3d = plot_cost_surface(X_b, y, theta_history)
        st.pyplot(fig3d)
    
    with viz_tab3:
        # Plot cost history
        fig2 = plot_cost_history(cost_history)
        st.pyplot(fig2)

    # Animate the gradient descent process
    if st.button("Animate Gradient Descent"):
        # Create full-width container for visualizations
        viz_container = st.container()
        
        # Create tabs for animation views
        anim_tab1, anim_tab2 = st.tabs(["2D Animation", "3D Animation"])
        plot_placeholder_2d = anim_tab1.empty()
        plot_placeholder_3d = anim_tab2.empty()
        
        # Create single table placeholder at the bottom
        st.subheader("Iteration History")
        table_placeholder = st.empty()
        
        progress_bar = st.progress(0)
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
            
            with viz_container:
                # Update 2D visualization
                fig_2d = plot_data_and_line(X, y, current_theta, 
                                        f"Iteration {i+1}", 
                                        iteration_data=iteration_data)
                plot_placeholder_2d.pyplot(fig_2d)
                
                # Update 3D visualization
                fig_3d = plot_cost_surface(X_b, y, theta_history[:i+1])
                plot_placeholder_3d.pyplot(fig_3d)
            
            # Update the single table placeholder
            table_placeholder.dataframe(
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
