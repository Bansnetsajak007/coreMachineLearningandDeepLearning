# 🏠 Sales using Linear Regression from Scratch

This project demonstrates how to build a simple **Linear Regression** model **from scratch** in Python using **Pandas** and **Numpy**, without using any machine learning libraries. The model is trained using **Gradient Descent** to predict **hMonthly Sales** based on the **Five Features**.

---

## 📁 Dataset Feature Description – Monthly Sales Dataset

This dataset simulates the monthly sales performance of a **single product** across different time periods or markets. It includes five features that influence the number of units sold (`monthly_sales`), which is the target variable.

### 🔍 Features:

- **`ad_spend`**  
  Monthly advertising budget (in $1000s). Higher ad spend typically boosts sales.

- **`product_price`**  
  Selling price of one unit of the product (in $) in 300 months. It can vary due to promotions, seasonality, or market testing and discounts. The company is constantly adjusting the price of their same product for various business reasons.

- **`market_trend_index`**  
  Index reflecting overall market demand (scaled from 40 to 100). A higher value means stronger demand of customer.

- **`seasonality_index`**  
  Seasonal multiplier (range: 0.5 to 1.5). Shows how seasonal effects boost or reduce sales. 1.0 is normal, above 1.0 is a seasonal boost, below 1.0 is a seasonal dip.
  eg ==> Like how ice cream sells more in summer or coats sell more in winter

- **`social_media_mentions`**  
  Total number of times the product was mentioned on social platforms in that month. More mentions often indicate higher customer awareness or virality.

### 🎯 Target:

- **`monthly_sales`**  
  The number of units sold in a given month. This is the value you're predicting in a regression model.


> Example row:  
> `2104, 399900`

## 📊 Objective

- Implement Linear Regression using only NumPy (no sklearn or ML libraries)
- Use Gradient Descent to optimize parameters (weights and bias)
- Visualize the loss (RSS), learned regression line, and predictions vs actual values

---

**Linear Regression Formula**  
y = b0 + w1*x1 + w2*x2 + w3 * x3 + w4 * x4 + w5 * x5

Where:  
ŷ: Predicted sales
x₁: Advertising spend (ad_spend)
x₂: Product price (product_price)
x₃: Market trend index (market_trend_index)
x₄: Seasonality index (seasonality_index)
x₅: Social media mentions (social_media_mentions)
w₁ to w₅: Weights (slopes) learned for each feature
b₀: Bias (intercept)

---

## 🚀 How It Works

### 1. **Preprocessing**
- Input features and targets are **standardized** (scaled to mean 0, std 1)

### 2. **Initialize Parameters**
- Start with `b0 = 0`, `w1 = 0` , `w2 = 0`, `w3 = 0`, `w4 = 0`, `w5 = 0`
- Define learning rate and number of epochs

### 3. **Training**
- For each epoch:
  - Predict sales using current weights
  - Calculate errors
  - Compute gradients
  - Update parameters using gradient descent

