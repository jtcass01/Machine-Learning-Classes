import numpy as np
import matplotlib.pyplot as plt

# load the data
X = []
Y = []
for line in open('data_poly.csv'):
    x,y = line.split(',')
    x = float(x)
    X.append([1, x, x*x]) #constant term, x term, x squared term
    Y.append(float(y))

# convert to Numpy arrays
X = np.array(X)
Y = np.array(Y)

# Let's plot to see what it looks like
plt.scatter(X[:,1], Y)
plt.show()

# calculate weights
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X,w)

# Plot again to make sure it works
plt.scatter(X[:,1], Y)
plt.plot(sorted(X[:,1]), sorted(Yhat)) # The sort function is needed or a bunch
# of lines will be created.  This is possible because a quadratic function is
# always monotonically increasing
plt.show()

# calculate the R-squared
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1)/d2.dot(d2)
print("r-squared:", r2)
