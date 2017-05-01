from __future__ import print_function, division
from builtins import range
import numpy as np
import pandas as pd


#M1 means the input size, M2 means the output size
def init_weight_and_bias(M1, M2):
    W = np.random.randn(M1, M2) / np.sqrt(M1)
    b = np.zeros(M2)
    #Must be turned into float 32s so we can use these in theano and tensorflow
    #without any complaints.
    return W.astype(np.float32), b.astype(np.float32)


# This function is used for convulusional networks
# Shape is four dimensional tuples
def init_filter(shape, poolsz):
    w = np.random.randn(*shape) / np.sqrt(np.prod(shape[1:]) + shape[0]*np.prod(shape[2:] / np.prod(poolsz)))
    return w.astype(np.float32)

# This is the rectifier which is used for an activiation function inside the neural
# network
def relu(x):
    return x * (x > 0)


def sigmoid(A):
    return 1 / (1 + np.exp(-A))

# Discussed in deep learning part 1
def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

# Calculates the cross-entropy error function
def sigmoid_cost(T, Y):
    return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()

# More general cross-entropy function that works with softmax
def cost(T, Y):
    return -(T*np.log(Y)).sum()

# Also calculates soft-max cross-entropy but does it in a fancy way
# Gives same answer as previous
def cost2(T, Y):
    # same as cost(), just uses the targets to index Y
    # instead of multiplying by a large indicator matrix with mostly 0s
    N = len(T)
    return -np.log(Y[np.arange(N), T]).mean()

# Gives us the error rate between the targets and predictions
def error_rate(targets, predictions):
    return np.mean(targets != predictions)

# Turns an Nx1 vector of targets into an indicator matrix which only has the values
# 0 and 1 of size Nxk
def y2indicator(y):
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

# Built to get all data from all classes.
def getData(balance_ones=True):
    # images are 48x48 = 2304 size vectors
    # N = 35887
    Y = []
    X = []
    first = True
    for line in open('fer2013/fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            # First collumn is a label
            Y.append(int(row[0]))
            # second column is space separated integers
            X.append([int(p) for p in row[1].split()])

    # Convert these to a numpy array and normalize the data
    X, Y = np.array(X) / 255.0, np.array(Y)

    # we lengthen class 1 by repeating it 9 times.
    if balance_ones:
        # balance the 1 class
        X0, Y0 = X[Y!=1, :], Y[Y!=1]
        # set x1 to be the samples where y = 1
        X1 = X[Y==1, :]
        # repeat 9 times
        X1 = np.repeat(X1, 9, axis=0)
        # restack
        X = np.vstack([X0, X1])
        Y = np.concatenate((Y0, [1]*len(X1)))

    return X, Y

# function we will use when we talk about convulutional neural networks
# Keeps the image shape.
def getImageData():
    X, Y = getData()
    N, D = X.shape
    d = int(np.sqrt(D))
    # n samples, one color channel, width which is d and the height which is d
    X = X.reshape(N, 1, d, d)
    return X, Y

# does same as getData but we only add the samples for which the class is zero
# or one.
def getBinaryData():
    Y = []
    X = []
    first = True
    for line in open('fer2013/fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            y = int(row[0])
            if y == 0 or y == 1:
                Y.append(y)
                X.append([int(p) for p in row[1].split()])
    return np.array(X) / 255.0, np.array(Y)


def crossValidation(model, X, Y, K=5):
    # split data into K parts
    X, Y = shuffle(X, Y)
    sz = len(Y) // K
    errors = []
    for k in range(K):
        xtr = np.concatenate([ X[:k*sz, :], X[(k*sz + sz):, :] ])
        ytr = np.concatenate([ Y[:k*sz], Y[(k*sz + sz):] ])
        xte = X[k*sz:(k*sz + sz), :]
        yte = Y[k*sz:(k*sz + sz)]

        model.fit(xtr, ytr)
        err = model.score(xte, yte)
        errors.append(err)
    print("errors:", errors)
    return np.mean(errors)
