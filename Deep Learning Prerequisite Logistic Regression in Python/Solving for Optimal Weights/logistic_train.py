import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from process import get_binary_data

X, Y = get_binary_data()
print("Pre-shuffle X: ", X)
print("Pre-shuffle Y: ", Y)
X, Y = shuffle(X, Y)
print("Post-shuffle X: ", X)
print("Post-shuffle Y: ", Y)

Xtrain = X[:-100]
Ytrain = Y[:-100]

Xtest = X[-100:]
Ytest = Y[-100:]

D = X.shape[1]
W = np.random.randn(D)
b = 0

def sigmoid(z):
    return 1 / (1+np.exp(-z))

def forward(X, W, b):
    return sigmoid(X.dot(W) + b)

def classification_rate(Y, P):
    return np.mean(Y == P)

# pY really means P(Y|X)
def cross_entropy(T, pY):
    return -np.mean(T*np.log(pY) + (1-T)*np.log(1-pY))

train_costs = []
test_costs = []
learning_rate = 0.001
for i in range(0,10000):
    pYtrain = forward(Xtrain, W, b)
    pYtest = forward(Xtest, W, b)

    ctrain = cross_entropy(Ytrain, pYtrain)
    ctest = cross_entropy(Ytest, pYtest)
    train_costs.append(ctrain)
    test_costs.append(ctest)

    W -= learning_rate*Xtrain.T.dot(pYtrain - Ytrain)
    b -= learning_rate*(pYtrain-Ytrain).sum()

    if i % 1000 == 0:
        print("i", i)
        print("ctrain", ctrain)
        print("ctest", ctest)

print("Final train classification_rate", classification_rate(Ytrain, np.round(pYtrain)))
print("Final test classification_rate", classification_rate(Ytest, np.round(pYtest)))

legend1, = plt.plot(train_costs, label = "train costs")
legend2, = plt.plot(test_costs, label= "test cost")
plt.legend([legend1, legend2])
plt.show()
