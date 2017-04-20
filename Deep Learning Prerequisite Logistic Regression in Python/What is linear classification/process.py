import numpy as np
import pandas as pd

def get_data():
    #load the data
    df = pd.read_csv('ecommerce_data.csv')
    data = df.as_matrix

    # split out X and Y, Y is the last column,  X is everything up to the last column
    X = data[:,:-1]
    Y = data[:,-1]

    # Normalize the numerical columns
    X[:,1] = normalize(X[:,1])
    X[:,2] = normalize(X[:,2])

    # Now we want to work on the categorical column.
    N, D = X.shape
    # We know this has to be shaped Nx(D+3) since we have four different categorical values.
    X2 = np.zeros((N, D+3))
    # We know that most of X is going to be same.
    X2[:,0:(D-1)] = X[:,0:(D-1)]

    # One hot encoding for the other four columns.
    for n in range(0, N):
        t = int(X[n, D-1])
        X2[n, t+D-1] = 1

    #Another way to do the one hot encoding
#    Z = np.zeros((N,4))
#    Z[np.arange(N), X[:,D-1].astype(np.int32)] = 1
    # X2[:,-4:] = Z
#    assert(np.abs(X2[:,-4:] - Z).sum() < 10e-10)

    return X2, Y

def get_binary_data():
    X, Y = get_data()

    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]

    return X2, Y2

def normalize(X):
    return (X - X.mean())/X.std()
