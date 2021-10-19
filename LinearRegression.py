import numpy as np
from scipy import linalg as sp_linalg

class LinearRegression(): 
    """A linear regression class.
    
    It takes in an X and y input, and calculates  
    the least squares coefficients and other model statistics."""

    def __init__(self, X, y): 
        """Initialize the class with X and y datasets."""
        self.X = X 
        self.y = y
        self.n = X.shape[1]
    
    def train(self):
        """Simple training using the normal equations.
        
        Requires calculating the inverse."""
        X = self.X
        y = self.y
        self.b_hat = np.linalg.inv(X.T @ X) @ X.T @ y

    def train_qr(self): 
        """QR decomposition based training.

        Using this method to decompose X into an orthogonal matrix Q, 
        and an upper triangular matrix R.  This simplifies to 
        R @ b_hat = Q.T @ y, which can be solved quickly via 
        backsubstitution."""
        # QR decomposition of X
        Q, R = np.linalg.qr(self.X, mode='reduced')
        # Solve Rb_hat = Q.Ty via backsubstitution (should be very fast)
        self.b_hat = sp_linalg.solve_triangular(R, Q.T @ self.y)

    def __str__(self):
        """Print out the first 7 columns and the coefficients.""" 
        print_string = ['X' + '\t' + 'b_hat' + '\t' + 'y']
        for i in range(min(len(self.y), 7)):
            try: 
                b_hat_clean = str(self.b_hat[i])
            except:
                b_hat_clean = ' '
            print_string.append(str(self.X[i]) + 
                                '\t' + b_hat_clean + 
                                '\t' + str(self.y[i]).strip())
        return '\n'.join(print_string)