"""Test out the algorithm on a real dataset.

data source source:
https://www.kaggle.com/quantbruce/real-estate-price-prediction/version/1

Use skikit-learn as a comparison
"""


import sys
sys.path.append('/Users/maxcomer/Dropbox/Python Deep Learning/linear_regression/linear_regression_github') #noqa
from  LinearRegression import LinearRegression
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

reg = LinearRegression(X, y,prepend_ones=True)

reg.train()
print(reg.b_hat)

reg.train_gd(lr=0.0000001, iterations=10, noisy=True)
print(reg.b_hat)



# Test against sklearn

print('\nsklearn results: ')

skreg = LinearRegressionSK()
skreg.fit(X, y)
print(skreg.coef_)
