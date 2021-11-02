"""Test out the algorithm on a real dataset.

data source:
https://www.kaggle.com/quantbruce/real-estate-price-prediction/version/1

Use skikit-learn as a comparison
"""


import sys
sys.path.append('/Users/maxcomer/Dropbox/Python Deep Learning/linear_regression/linear_regression_github')  # noqa
from LinearRegression import LinearRegression
from GDHelperFunctions import MSE
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as LinearRegressionSK


# Load in the real estate data
df = pd.read_csv('tests/real_estate.csv')
print(df.head())
print(df.info())

# Split out X and y datasets
X = df.drop(['No', 'Y house price of unit area'], axis=1).values
print(X.shape)

y = df['Y house price of unit area'].values.reshape(len(X), 1)
print(y.shape)

reg = LinearRegression(X, y, prepend_ones=True)

print('\nTrain using the standard approach')
reg.train()
print(reg.b_hat.shape)
print([round(b, 3) for b in reg.b_hat[:, 0]])
print('Regular MSE ', MSE(reg.X, y, reg.b_hat)[1])

# Gradient descent is producing different results based on the starting values
# of b_hat
print('\nTrain using gradient descent')
reg.train_gd(lr=0.0001, iterations=100000, noisy=True)
print([round(b, 3) for b in reg.b_hat[:, 0]])
print('GD MSE ', MSE(reg.X, y, reg.b_hat)[1])


# Test against sklearn

print('\nsklearn results: ')

skreg = LinearRegressionSK()
skreg.fit(X, y)
print(skreg.intercept_, skreg.coef_)
