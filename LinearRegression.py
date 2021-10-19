import numpy as np
import scipy as sp

class LinearRegression(): 
    """A linear regression class which takes in an X and y input, and calculates  
    the least squares coefficients and other model statistics."""
    # Initialize the class with X and y datasets 
    def __init__(self, X, y): 
        self.X = X 
        self.y = y
        self.n = X.shape[1]
    
    # Simple training using the normal equations derived through linear algebra 
    def train(self):
        X = self.X
        y = self.y
        self.b_hat = np.linalg.inv(X.T @ X) @ X.T @ y

    # QR decomposition based training 
    def train_qr(self): 
        # QR decomposition of X
        Q, R = np.linalg.qr(self.X, mode='reduced')
        # Solve Rb_hat = Q.Ty via backsubstitution (should be very fast)
        self.b_hat = sp.linalg.solve_triangular(R, Q.T @ self.y)

    def __str__(self):
        # Print out the first 7 columns and the coefficients 
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