import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# === Step 0: Load & Normalize Data === #
x1 = np.array([8, 10, 12, 14, 16, 18, 20, 22, 24, 26], dtype=float)
x2 = np.array([2, 2, 3, 3, 4, 4, 5, 5, 6, 6], dtype=float)
y = np.array([35, 45, 58, 69, 81, 93, 105, 117, 129, 141], dtype=float)

x1_ready = (x1 - x1.mean()) / x1.std()
x2_ready = (x2 - x2.mean()) / x2.std()

n = len(y)

# === Step 1: Initialize Parameters === #
b0 = 0
w1 = 0
w2 = 0
alpha = 0.001
epochs = 1000

# For animation: Save history of weights
history = []

# === Step 2: Gradient Descent === #
for epoch in range(epochs):
    y_pred = b0 + w1 * x1_ready + w2 * x2_ready
    error = y - y_pred

    db0 = -2 * np.sum(error)
    dw1 = -2 * np.sum(error * x1_ready)
    dw2 = -2 * np.sum(error * x2_ready)

    b0 -= alpha * db0
    w1 -= alpha * dw1
    w2 -= alpha * dw2

    if epoch % 10 == 0:
        history.append((b0, w1, w2))  # Save every 10th frame

# === Step 3: Animation Setup === #
fig, ax = plt.subplots()
ax.set_title("Gradient Descent: Regression Line Animation")
ax.set_xlabel("Area of House (normalized)")
ax.set_ylabel("Price ($1000s)")
sc = ax.scatter(x1_ready, y, label="Data Points")
line, = ax.plot([], [], 'r-', label="Regression Line")
ax.legend()

# Fix x2 at 0 (its mean after normalization)
x_vals = np.linspace(x1_ready.min(), x1_ready.max(), 100)

def animate(i):
    b0, w1, w2 = history[i]
    y_vals = b0 + w1 * x_vals  # x2 is 0
    line.set_data(x_vals, y_vals)
    return line,

ani = animation.FuncAnimation(fig, animate, frames=len(history), interval=100, blit=True)

plt.show()
