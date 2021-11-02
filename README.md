## Linear regression from scratch

This is a project to build a basic linear regression model algorithm from
scratch using numpy and scipy libraries, and implementing three different
training methods: the standard linear algebra method X'XX'y, a version of 
this using QR decomposition to test performance, and a gradient descent 
algorithm.  


**Setup**

This requires the following libraries:
numpy
scipy
warnings

Running the test scripts further require:
pandas
sys
timeit
random
sklearn (to test as a comparison)

**How to run**

The LinearRegression object takes in X and y arrays upon instantiation,
X representing a nxm numpy array with n observations and m independent
variables.  y represents the dependent variable and is a nx1 array.

By default the algorithm will prepend a column of 1s for the intercept
to the X value and standardize the values of X to make gradient descent
work correctly.  

To run the test scripts correctly, you may need to update the sys.path 
reference to work with your local directory.  