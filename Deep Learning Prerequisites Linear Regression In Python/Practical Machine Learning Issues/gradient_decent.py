import numpy as np
import matplotlib.pyplot as plt

# number of data points
N = 10
# Dimensionality
D = 3

# Initialize an NxD matrix
X = np.zeros((N,D))

# Set the bias term
X[:,0] = 1

# Set the first five elements of the first column to one, last 5 of second column to one
X[:5,1] = 1
X[5:,2] = 1

#Set the targets to be 0 for the first half of the data and 1 for the second half
Y = np.array([0]*5 + [1]*5)

# Proof that regular solution does not work out.  Returns error "singular matrix"
#w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))

# Use gradient decent instead
costs = []

# Initialize random weights, ensured that it has variance of 1/D
w = np.random.randn(D) / np.sqrt(D)
learning_rate = .001

for t in range(0,500):
    Yhat = X.dot(w)
    delta = Yhat-Y
    w = w- learning_rate*X.T.dot(delta)
#    print("w",w)


    mse = delta.dot(delta) / N
    costs.append(mse)

plt.plot(costs)
plt.show()

print("X", X)
print("Yhat", Yhat)
print("Y", Y)
#print("costs", costs)

plt.plot(Yhat, label='prediction')
plt.plot(Y, label='target')
plt.legend()
plt.show()
