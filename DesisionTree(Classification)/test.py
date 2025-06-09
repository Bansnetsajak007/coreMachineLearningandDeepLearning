import pandas as pd 
import numpy as np


def calculate_gini_impurity(features, target):
    pass

data_Set = pd.DataFrame({
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female'],
    'Age': [23, 45, 34, 23, 54],
    'Bought': ['Yes', 'No', 'Yes', 'No', 'Yes']
})

features = data_Set.drop(columns=['Bought'])
target = data_Set['Bought']
gini_value = calculate_gini_impurity(features, target)
print(f"Gini Impurity: {gini_value:.4f}")

