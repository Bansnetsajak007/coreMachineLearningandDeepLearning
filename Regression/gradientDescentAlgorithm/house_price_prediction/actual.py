import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Step 0: Load and Prepare Data === #
housing_data = pd.read_csv('house-price-prediction-using-linear-regression.txt', usecols=[0, 2], names=['size', 'price'])

# Extract raw values
X_raw = housing_data['size'].values
Y_raw = housing_data['price'].values

# Standardize the data
X_mean = X_raw.mean()
X_std = X_raw.std()
Y_mean = Y_raw.mean()
Y_std = Y_raw.std()

X = (X_raw - X_mean) / X_std
Y = (Y_raw - Y_mean) / Y_std

# === Step 1: Initialize Parameters === #
b0 = 0     # Intercept
w1 = 0     # Weight for size (x)
learning_rate = 0.01
epochs = 1000

# === Step 2: Gradient Descent === #
for epoch in range(epochs):
    y_pred = b0 + w1 * X
    errors = Y - y_pred

    # Compute gradients
    db0 = -2 * np.sum(errors)
    dw1 = -2 * np.sum(errors * X)

    # Update parameters
    b0 -= learning_rate * db0
    w1 -= learning_rate * dw1

    # Print RSS every 100 steps
    if epoch % 100 == 0:
        rss = np.mean(errors ** 2)
        print(f"Epoch {epoch}: RSS = {rss:.4f} | b0 = {b0:.4f}, w1 = {w1:.4f}")

# === Step 3: Final Parameters === #
print("\n🏁 Training completed!")
print(f"Final RSS = {rss:.4f}")
print(f"Intercept (b0) = {b0:.4f}")
print(f"Weight (w1) = {w1:.4f}")

# === Step 4: Plot - Standardized Data === #
Y_pred = b0 + w1 * X

plt.figure(figsize=(8, 5))
plt.scatter(X, Y, color='blue', label='Actual (Standardized)')
plt.plot(X, Y_pred, color='red', label='Predicted Line', linewidth=2)
plt.xlabel('Size (Standardized)')
plt.ylabel('Price (Standardized)')
plt.title('Linear Regression Fit (Standardized)')
plt.legend()
plt.grid(True)
plt.show()

# === Step 5: Plot - Original Scale === #
# Convert standardized predictions back to original scale
X_pred_raw = X * X_std + X_mean
Y_pred_raw = Y_pred * Y_std + Y_mean

plt.figure(figsize=(8, 5))
plt.scatter(X_raw, Y_raw, color='blue', label='Actual Price')
plt.plot(X_pred_raw, Y_pred_raw, color='red', label='Predicted Price Line', linewidth=2)
plt.xlabel('Size (sqft)')
plt.ylabel('Price ($)')
plt.title('Linear Regression Fit (Original Scale)')
plt.legend()
plt.grid(True)
plt.show()
