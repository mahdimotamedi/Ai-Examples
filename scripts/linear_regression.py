import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Set a random seed for reproducibility
np.random.seed(42)

# Generate synthetic data for linear regression
# X is the independent variable, y is the dependent variable
X = 2 * np.random.rand(100, 1)  # Random values between 0 and 2
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3X + noise

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"Model Coefficient: {model.coef_[0][0]:.2f}")
print(f"Model Intercept: {model.intercept_[0]:.2f}")

# Visualize the test data and the regression line
plt.scatter(X_test, y_test, color="blue", label="Actual Data")  # Scatter plot of actual test data
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Regression Line")  # Plot regression line
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression Model")
plt.show()
