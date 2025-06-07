# voli learn about loss function, how gradinet descent works, and how to implement logistic regression from scratch.

# logistic regression from stracth

# using data set with 1 feature 

'''
Income	Bought
2380	0
7351.1	0
48224.4	1
4833	0
18426.1	0
52709	1
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def sigmoid(z):
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-z))

def calculate_gradient(X, y, b, w1, learning_rate):
    # y = w1*x1 --> linear prediction
    # y = sigmoid(w1*x1 + w0) --> logistic prediction
    #where sigmoid(z) = 1 / (1 + np.exp(-z))
    alpha = learning_rate
    epochs = 1000

# optimizing the fucking parameters using GRADIENT
    for epoch in range(epochs):
        y_pred = sigmoid(w1 * X + b)  # linear prediction with sigmoid activation

        # error = - (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        error = y_pred - y 


        #derivative of the loss function with respect to w1 and b
        derivative_b = np.mean(error)
        derivative_w1 = np.mean(error * X)

        #update the parameters
        #step size 
        step_size_b = alpha * derivative_b
        step_size_w1 = alpha * derivative_w1

        b = b - step_size_b
        w1 = w1 - step_size_w1

    return b, w1,y_pred

def predict(X, b, w1):
    """Make predictions using the logistic regression model."""
    z = w1 * X + b
    return sigmoid(z)

income_data = pd.read_csv('./Product_sales.csv')

X = income_data['Income'].values
X_normalize = (X - X.mean()) / X.std() # overwrites X with normalized values

y = income_data['Bought'].values

bias, weight, y_pred = calculate_gradient(X_normalize, y, 0, 0, 0.001)

# print(f"Bias: {bias}")
# print(f"Weight: {weight}")
#calculating accuracy of model
cm = confusion_matrix(y, (y_pred > 0.5).astype(int))
print("Confusion Matrix:\n", cm)

accuracy = (cm[0, 0] + cm[1, 1]) / (cm[0,0] + cm[0, 1] + cm[1, 0] + cm[1, 1])
print("Accuracy:", accuracy)
# mean = X.mean()
# std = X.std()

# # Normalize test samples
# test_samples = np.array([4000, 5000, 6000, 7000, 8000, 39000])
# test_samples = (test_samples - mean) / std

# # Now make prediction
# predictions = predict(test_samples, bias, weight)
# binary_preds = (predictions > 0.5).astype(int)
# print("Raw Prediction:", predictions)
# print("Predicted classes:", binary_preds)


