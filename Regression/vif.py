import pandas as pd
import numpy as np

# detecting multicollinearity --> variance inflation factor (VIF)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def calculate_vif(X):
    x_columns = X.columns
    for i in range (0, x_columns.shape[0]):
        y = x_columns[i]
        x_others = x_columns.drop(x_columns[i])
        linear = LinearRegression()
        linear.fit(X[x_others], X[y])
        rsq = r2_score(X[y], linear.predict(X[x_others]))
        try:
            vif = round(1 / (1 - rsq), 2)
        except ZeroDivisionError:
            vif = float("inf")
        print(y, "VIF :", vif)



if __name__ == "__main__":
    # Load the dataset

    sales_df = pd.read_csv('./monthly_sales_dataset.csv')
    sales_df['half_price'] = sales_df['product_price'] / 2

    X = sales_df.drop(columns = ['monthly_sales'])

    calculate_vif(X)

'''
output:
ad_spend VIF : 1.02
product_price VIF : inf
market_trend_index VIF : 1.01
seasonality_index VIF : 1.01
social_media_mentions VIF : 1.01
half_price VIF : inf

VIF = 1 / (1 - R^2)

inf (infinity) means that a variable is perfectly linearly dependent on one or more other variables.
when R^2 = 1, then VIF is infinite, indicating perfect multicollinearity. meaning feature can be perfectly predicted by other features.

'''