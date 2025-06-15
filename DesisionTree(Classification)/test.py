# i didn't get anything hyaaa feri suru dekhi herxu 
import pandas as pd 
import numpy as np


def gini_impurity(labels):
    probs = labels.value_counts(normalize=True)
    return 1 - np.sum(np.square(probs))

def gini_index_split(data, feature, target):
    values = data[feature].unique()
    gini = 0
    total_len = len(data)
    
    for val in values:
        subset = data[data[feature] == val]
        weight = len(subset) / total_len
        impurity = gini_impurity(subset[target])
        gini += weight * impurity
        
    return gini

def best_split(data, features, target):
    best_feature = None
    min_gini = float('inf')
    
    for feature in features:
        gini = gini_index_split(data, feature, target)
        if gini < min_gini:
            min_gini = gini
            best_feature = feature
            
    return best_feature, min_gini
def build_tree(data, features, target):
    # If all labels are same, return leaf node
    if len(data[target].unique()) == 1:
        return data[target].iloc[0]

    # If no features left to split
    if len(features) == 0:
        return data[target].mode()[0]  # majority class

    # Choose best feature to split
    best_feat, _ = best_split(data, features, target)
    tree = {best_feat: {}}

    # Recurse for each value in best feature
    for val in data[best_feat].unique():
        subset = data[data[best_feat] == val]
        subtree = build_tree(subset, [f for f in features if f != best_feat], target)
        tree[best_feat][val] = subtree

    return tree


data_Set = pd.DataFrame({
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female'],
    'Income Band': ['High', 'High', 'Low', 'Low', 'High'],
    'Bought': ['Yes', 'No', 'Yes', 'No', 'Yes']
})

features = ['Gender', 'Income Band']
target = 'Bought'

tree = build_tree(data_Set, features, target)
print(tree)


