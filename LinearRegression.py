"""Implement a linreg class with multiple training methods."""


import numpy as np
from scipy import linalg as sp_linalg
from GDHelperFunctions import MSE, MSEGradient, Standardize


class LinearRegression():
    """A linear regression class.

    It takes in an X and y input, and calculates
    the least squares coefficients and other model statistics.
    """

    def __init__(self, X, y, prepend_ones=True, standardize=True):
        """Initialize the class with X and y datasets.

        Also prepend a column of ones if requested.
        """
        self.X = X
        self.y = y
        self.n = X.shape[0]

        if len(set(X[:, 0])) == 1:
            raise Exception('Do not include an intercept column')

        if standardize:
            self.X = Standardize(self.X)

        if prepend_ones:
            ones = np.ones((self.n, 1))
            self.X = np.concatenate((ones, self.X), axis=1)

    def train(self):
        """Train using the normal equations and inverse.

        Requires calculating the inverse.
        """
        X = self.X
        y = self.y
        self.b_hat = np.linalg.inv(X.T @ X) @ X.T @ y

    def train_qr(self):
        """Train using QR decomposition to compare performance.

        Using this method to decompose X into an orthogonal matrix Q,
        and an upper triangular matrix R.  This simplifies to
        R @ b_hat = Q.T @ y, which can be solved quickly via
        backsubstitution.
        """
        # QR decomposition of X
        Q, R = np.linalg.qr(self.X, mode='reduced')
        # Solve Rb_hat = Q.Ty via backsubstitution (should be very fast)
        self.b_hat = sp_linalg.solve_triangular(R, Q.T @ self.y)

    def train_gd(self, lr=0.01, iterations=100, noisy=True):
        """Use gradient descent to train the model.

        Initialize b_hat with random values.
        mse: loss function - mean squared error or (y - y_hat)**2
        d_b_hat: derivative of the loss function wrt b_hat (includes intercept)
        MSE and Gradient calcutaions are handled in the separate helper file.
        The algorithm will raise an exception if MSE is not converging.
        """
        self.b_hat = np.zeros((self.X.shape[1], 1))
        # self.b_hat = np.random.randn(self.X.shape[1], 1)

        # Gradient descent
        last_mse = 0
        mult = iterations / 10

        for i in range(iterations):
            y_hat, mse = MSE(self.X, self.y, self.b_hat)

            if last_mse < mse and last_mse != 0:
                raise Exception('MSE increasing, use a lower learning rate')

            # Get the derivative of the MSE wrt b_hat and apply
            d_b_hat = MSEGradient(self.X, self.y, y_hat)
            if noisy and i % mult == 0:
                print(f'\nIteration {i}, mse: ',  mse)
                # print('b_hat: ', self.b_hat.T)
                # print('d_b_hat: ',d_b_hat.T)

            self.b_hat -= d_b_hat * lr

            last_mse = mse

    def __str__(self):
        """Print out the first 7 columns and the coefficients."""
        print_string = ['X' + '\t' + 'y' + '\t' + 'b_hat']
        for i in range(min(len(self.y), 7)):
            try:
                b_hat_clean = str(self.b_hat[i])
            except(IndexError):
                b_hat_clean = ' '
            print_string.append(str(self.X[i]) +
                                '\t' + str(self.y[i]).strip() +
                                '\t' + b_hat_clean)
        return '\n'.join(print_string)
