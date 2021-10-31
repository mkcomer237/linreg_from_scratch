"""Implement necesssary helper functions for gradient descent."""


import numpy as np
import warnings
warnings.filterwarnings("error")

def MSE(X, y, b_hat):
    """Return y_hat and mean squared error in a tuple."""
    y_hat = X @ b_hat
    e = y - y_hat
    try:
        mse = ((e.T @ e)/len(y))[0, 0]
    except(RuntimeWarning):
        raise Exception(f'Convergence error, latest e: {e[0]}')
    return (y_hat, mse)

def MSEGradient(X, y, y_hat):
    """Find the array of derivatives for MSE wrt b_hat."""
    e = y - y_hat
    return -2*(X.T @ e)/len(y)
