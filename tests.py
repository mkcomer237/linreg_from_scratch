"""Script to test the various training methods on data."""

from  LinearRegression import LinearRegression
import numpy as np
from numpy.random import normal
import random
from timeit import default_timer as timer

X = np.array([[1, 1, 1, 1], [0, 1, 3, 4]]).T
y = np.array([0, 8, 8, 20]).reshape(4, 1)

print('Small example')

print('Train using X.T@X inverse')
reg = LinearRegression(X, y)
reg.train()
print(reg)

print('Train using QR decomposition')
reg.train_qr()
print(reg)


# Generate a large random dataset totest speed
print('\nLarge random data example to test speed')

obs = 100000

x0 = [1 for i in range(obs)]
x1 = list(range(obs))
x2 = list(range(obs))
random.shuffle(x2)
y = [normal(3)*x1 + normal(2)*x2 + normal(5) 
     for x1, x2 in zip(x1, x2)]

X = np.hstack((np.array(x0).reshape(obs, 1),
               np.array(x1).reshape(obs, 1), 
               np.array(x2).reshape(obs, 1)))

reg = LinearRegression(X, y)

# Time each approach
start = timer()
reg.train()
end = timer()
print('Training using inverse:', end - start) # Time in seconds, e.g. 5.38091952400282

start = timer()
reg.train_qr()
end = timer()
print('Training using QR:', end - start) # Time in seconds, e.g. 5.38091952400282
