import numpy as np
from process import get_binary_data

X, Y = get_binary_data()

# Get the dimensionality of the data set
D = X.shape[1]
# Use the dimensionality to initialize the weights of the regression model
W = np.random.randn(D)
# b is the bias term.
b = 0


def sigmoid(a):
    return 1 /  (1+np.exp(-a))

def forward(X, W, b):
    return sigmoid(X.dot(W) + b)

# P of Y given X P(Y|X)
P_Y_given_X = forward(X, W, b)
predictions = np.round(P_Y_given_X)

def classification_rate(Y, P):
    # looks like it would return an array of booleans but it actually divides the
    # number of correct by the total number
    return np.mean(Y == P)

print("Score:", classification_rate(Y, predictions))

# Next we will look at how to train these weights so we can get this score more
# accurate.
