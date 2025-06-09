import pandas as pd
import numpy as np


def gini_impurity(features, target):
    #gini impurity --> # 1 - sum(p_i^2) where p_i is the proportion of class i in the target variable
    pass
def information_gain():
    pass




df = pd.read_csv('./cleaned_call_center_survey.csv')
features  = df.drop(columns=['Overall_Satisfaction'])
target = df['Overall_Satisfaction']




