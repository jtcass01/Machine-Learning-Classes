import numpy as np
import matplotlib.pyplot as plt

# number of samples
N = 100
# Dimension of array
D = 2

# DxN normally distributed data matrix
X = np.random.randn(N,D)
# Add a column of ones and include the bias term in the weights w
ones = np.array([[1]*N]).T
# Concatenate the two arrays..
Xb = np.concatenate((ones, X), axis=1)

# randomly intialize a weight vector of length D +1.  The values don't matter
# right now since we don't have any labels and just want to calculate the sigmoid
w = np.random.randn(D+1)

# Calculate the weights by multiplying everything in Xb by that in w.
z = Xb.dot(w)
# This gives us an n by 1 vector

def sigmoid(z):
    return 1/(1+np.exp(-z))

# As you can see, our values are between 0 and 1 as expected.
print(sigmoid(z))


#Additional just for fun
plt.plot(sorted(sigmoid(z)), label = "Sorted sigmoid")
plt.plot(sigmoid(z), label = "Raw sigmoid")
plt.legend()
plt.title("Sigmoid practice")
plt.show()
