"""Implement any necesssary loss functions."""


import numpy as np


def MSE(X, y, b_hat):
    """return y_hat and mean squared error in a tuple."""
    y_hat = y - (X @ b_hat)
    e = y - y_hat
    mse = ((e.T @ e)/len(y))[0, 0]
    return (y_hat, mse)

#def d_b_hat():
