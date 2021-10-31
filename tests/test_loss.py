"""Smaller set of tests for gradient descent."""

import sys
sys.path.append('/Users/maxcomer/Dropbox/Python Deep Learning/linear_regression/linear_regression_github') #noqa
print(sys.path)
from GDHelperFunctions import MSE, MSEGradient
from LinearRegression import LinearRegression
import numpy as np


X = np.array([0, 1, 3, 4]).reshape(4, 1)
y = np.array([0, 8, 8, 20]).reshape(4, 1)
#b_hat = np.array([2, 3]).reshape(2, 1)
#print('Shapes (X, b_hat):', X.shape, b_hat.shape)


#print(MSE(X, y, b_hat))

reg = LinearRegression(X, y, prepend_ones=True)

reg.train_gd(noisy=False)
print(reg.b_hat)