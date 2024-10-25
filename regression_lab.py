import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = pd.read_csv('Nairobi_Office_Price_Ex.csv')

# Extract the features (office size) and target (price)
X = data['SIZE'].values
y = data['PRICE'].values

# Normalize the features (X)
X_mean = np.mean(X)
X_std = np.std(X)
X_normalized = (X - X_mean) / X_std  # Standardizing the 'SIZE' feature

# Step 2: Define the Mean Squared Error function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Step 3: Define the Gradient Descent function
def gradient_descent(X, y, m, c, learning_rate, epochs):
    n = float(len(y))  # Number of data points
    for epoch in range(epochs):
        # Make predictions
        y_pred = m * X + c
        
        # Calculate Mean Squared Error
        error = mean_squared_error(y, y_pred)
        
        # Compute gradients for slope (m) and intercept (c)
        D_m = (-2/n) * np.dot(X, (y - y_pred))  # Gradient with respect to m
        D_c = (-2/n) * np.sum(y - y_pred)      # Gradient with respect to c
        
        # Update the slope (m) and intercept (c)
        m -= learning_rate * D_m
        c -= learning_rate * D_c
        
        # Print the error at each epoch
        print(f'Epoch {epoch + 1}: Error = {error}')
        
        # Early stopping: break if error is low enough
        if error < 1e-5:
            print("Early stopping triggered.")
            break
    
    return m, c

# Step 4: Train the model using Gradient Descent
# Initialize values for slope (m) and intercept (c) randomly
np.random.seed(0)  # For reproducibility
m = np.random.rand()  # Random initial value for slope
c = np.random.rand()  # Random initial value for intercept

# Define the learning rate and number of epochs
learning_rate = 0.1  # Increased learning rate for faster convergence
epochs = 10  # Number of epochs for training

# Train the model
m, c = gradient_descent(X_normalized, y, m, c, learning_rate, epochs)

# After training, print the final values of m and c
print(f'Final slope (m): {m}')
print(f'Final intercept (c): {c}')

# Step 5: Plot the Line of Best Fit
# Scatter plot of the data
plt.scatter(X, y, color='blue', label='Data points')

# Line of best fit
y_pred = m * X_normalized + c
plt.plot(X, y_pred, color='red', label='Line of Best Fit')

# Add labels and legend
plt.xlabel('Office Size (sq. ft)')
plt.ylabel('Office Price')
plt.legend()

# Show the plot
plt.title("Office Size vs Price with Line of Best Fit")
plt.show()

# Step 6: Predict Office Price for 100 sq. ft
office_size = 100
office_size_normalized = (office_size - X_mean) / X_std
predicted_price = m * office_size_normalized + c
print(f'Predicted price for 100 sq. ft office: {predicted_price}')
