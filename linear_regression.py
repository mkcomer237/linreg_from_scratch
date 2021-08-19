import numpy as np

class linear_regression(): 
    """A linear regression class which takes in an X and y input, and calculates the 
    least squares coefficients and other model statistics."""
    # Initialize the class with X and y datasets 
    def __init__(self, X, y): 
        self.X = X 
        self.y = y
    
    # Simple training using the normal equations derived through linear algebra 
    def train(self):
        X = self.X
        y = self.y
        self.b_hat = np.linalg.inv(X.T @ X) @ X.T @ y