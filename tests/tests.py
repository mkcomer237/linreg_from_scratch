"""Script to test the various training methods on data."""


import sys
sys.path.append('/Users/maxcomer/Dropbox/Python Deep Learning/linear_regression/linear_regression_github')  # noqa
from LinearRegression import LinearRegression
import numpy as np
from numpy.random import normal
import random
from timeit import default_timer as timer

X = np.array([[0, 1, 3, 4, 3], [3, 2, 3, 5, 9]]).T
y = np.array([0, 8, 8, 20, 15]).reshape(5, 1)

print('Small example')

print('\nTrain using X.T@X inverse')
reg = LinearRegression(X, y)
reg.train()
print(reg)

print('\nTrain using QR decomposition')
reg.train_qr()
print(reg)
print('b_hat shape: ', reg.b_hat.shape)

print('\nTrain using Gradient Descent')
reg.train_gd(lr=0.001, noisy=False, iterations=10000)
print(reg)
print('b_hat shape: ', reg.b_hat.shape)


# Generate a large random dataset to test speed
print('\nLarge random data example to test speed')

obs = 1000

x1 = list(range(obs))
x2 = list(range(obs))
random.shuffle(x2)
y = np.array([normal(3)*x1 + normal(2)*x2 + normal(5)
              for x1, x2 in zip(x1, x2)]).reshape(obs, 1)

X = np.hstack((np.array(x1).reshape(obs, 1),
               np.array(x2).reshape(obs, 1)))

print('\nNew dataset shape: ', X.shape)

reg = LinearRegression(X, y)

# Time each approach
start = timer()
reg.train()
end = timer()
print('Training using inverse:', end - start, reg.b_hat)

start = timer()
reg.train_qr()
end = timer()
print('Training using QR:', end - start, reg.b_hat)

start = timer()
reg.train_gd(lr=0.001, noisy=True, iterations=5000)
end = timer()
print('Training using GD:', end - start, reg.b_hat)
