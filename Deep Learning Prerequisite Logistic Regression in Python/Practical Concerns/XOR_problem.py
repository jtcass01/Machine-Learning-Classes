import numpy as np
import matplotlib.pyplot as plt

N = 4
D = 2

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1],
    ])

T = np.array([0,1,1,0])

ones = np.array([[1]*N]).T

# Plot to understand the problem
#plt.scatter(X[:,0], X[:,1], c=T)
#plt.show()

# The trick to the XOR problem is to add another dimension to our dataset
xy = np.matrix(X[:,0]*X[:,1]).T
Xb = np.array(np.concatenate((ones, xy, X), axis=1))

# randomly initialize the weights
w = np.random.randn(D+2)

# calculate the model output
z = Xb.dot(w)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

Y = sigmoid(z)

# calculate the cross-entropy error
def cross_entropy(T,Y):
    E = 0
    for i in range(0,N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1-Y[i])
    return E

learning_rate = 0.005
error = []
for i in range(0,5000):
    e = cross_entropy(T,Y)
    error.append(e)

    if i % 100 == 0:
        print(e)

    # gradient descent weight update with regularization
    w += learning_rate * (Xb.dot(T-Y).T - 0.01*w)

    # recalculate Y
    Y = sigmoid(Xb.dot(w))

plt.plot(error)
plt.title("Cross-entropy per iteration")
plt.show()

print("Final w:", w)
print("Final classification rate:", 1-np.abs(T-np.round(Y)).sum() / N)
