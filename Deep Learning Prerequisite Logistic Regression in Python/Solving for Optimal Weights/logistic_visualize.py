import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 2

X = np.random.randn(N,D)

# center the first 50 points at (-2,-2)
X[:50,:] = X[:50,:] - 2*np.ones((50,D))
# center the last 50 points at (2,2)
X[50:,:] = X[50:,:] + 2*np.ones((50,D))

# labels: first 50 are 0, last 50 are 1
T = np.array([0]*50 + [1]*50)

# add a column of ones
ones = np.array([[1]*N]).T
Xb = np.concatenate((ones, X), axis = 1)

def sigmoid(z):
    return 1 / (1*np.exp(-z))

# get the closed-form solution
w = np.array([0,4,4])

# should give us the line y = -x

# Set the color of our plots to the targets.  Set the size to 100
# Set the alpha to 0.5 so the dots are transparent
plt.scatter(X[:,0], X[:,1], c=T, s=100, alpha=0.5)
x_axis = np.linspace(-6,6, 100)
y_axis = -x_axis

plt.plot(x_axis, y_axis)
plt.show()
