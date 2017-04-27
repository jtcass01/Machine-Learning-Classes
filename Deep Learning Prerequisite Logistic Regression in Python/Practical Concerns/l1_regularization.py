import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1+np.exp(-z))

N = 50
D = 50

# Uniformly distributed numbers between -5 and +5
# Subtract 0.5 to center it around zero, then multiply by 10
X = (np.random.random((N,D)) - 0.5)*10

# Only the first three dimensions actually effect the output, the rest are zero
# so the last 47 dimensions do not effect the target at all
true_w = np.array([1,0.5,-0.5] + [0]*(D-3))

# Build Y, add some random noise.
Y = np.round(sigmoid(X.dot(true_w) + np.random.randn(N)*0.5))

# Perform gradient descent
costs = []
w = np.random.randn(D) / np.sqrt(D)
learning_rate = 0.001
# Try different values
l1 = 9.05

for t in range(0,5000):
    Yhat = sigmoid(X.dot(w))
    delta = Yhat - Y
    w = w - learning_rate*(X.T.dot(delta) + l1*np.sign(w))

    cost = -(Y*np.log(Yhat) + (1-Y)*np.log(1-Yhat)).mean() + l1*np.abs(w).mean()
    costs.append(cost)

plt.plot(costs)
plt.show()

plt.plot(true_w, label="true_w")
plt.plot(w, label="w map")
plt.legend()
plt.show()
