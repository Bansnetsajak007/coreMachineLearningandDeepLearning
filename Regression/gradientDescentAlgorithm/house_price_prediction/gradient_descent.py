# gradient descent algorithm for linear regression
# how paramaters are updated with aim to reduce cost function

import numpy as np 

# dataset 
# X1 --> Area of House (sq ft)
# X2 --> Number of Bedrooms
# y --> Price of House (in $1000s)
# x1 and x2 are features, y is the target variable
# x1 and x2 are standardized

x1 = np.array([8, 10, 12, 14, 16, 18, 20, 22, 24, 26], dtype=float)
x1_ready = (x1 - x1.mean()) / x1.std()
x2 = np.array([2, 2, 3, 3, 4, 4, 5, 5, 6, 6], dtype=float)
x2_ready = (x2 - x2.mean()) / x2.std()
y = np.array([35, 45, 58, 69, 81, 93, 105, 117, 129, 141], dtype=float)
n = len(y)  # Number of data points

# initialize parameters
b0 = 0     # Intercept
w1 = 0     # Weight for area_of_house (x1)
w2 = 0     # Weight for number_of_rooms (x2)
learning_rate = 0.001  # Learning rate
epochs = 1000   # Number of iterations

# optimizing parameters using gradient descent
for epoch in range(epochs):
    y_pred = b0 + w1 * x1_ready + w2 * x2_ready

  # computing errors
    errors = y - y_pred

  # computing gradients (derivatives of rss)
    db0 = -2 * np.sum(errors)
    dw1 = -2 * np.sum(errors * x1_ready)
    dw2 = -2 * np.sum(errors * x2_ready)

  # calculating step size from previous point what step to take
    step_size_intercept = db0 * learning_rate
    step_size_w1 = dw1 * learning_rate
    step_size_w2 = dw2 * learning_rate

    # calculating new parameters (updating parameters)
    b0 = b0 - step_size_intercept
    w1 = w1 - step_size_w1
    w2 = w2 - step_size_w2

    # checking parameters every 100 epochs
    if epoch % 100 == 0:
        rss = np.mean(errors ** 2) 
        print(f"Epoch {epoch}: RSS = {rss:.2f} | b0 = {b0:.2f}, w1 = {w1:.2f}, w2 = {w2:.2f}")

# final parameters after training
print("\n🏁 Training completed!")
print(f"Final parameters:")
print(f"RSS= {rss:.4f}")
print(f"b0 (intercept)       = {b0:.4f}")
print(f"w1 (area_of_house)   = {w1:.4f}")
print(f"w2 (number_of_rooms) = {w2:.4f}")