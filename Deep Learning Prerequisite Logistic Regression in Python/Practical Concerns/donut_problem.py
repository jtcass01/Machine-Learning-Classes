import numpy as np
import matplotlib.pyplot as plt

N = int(1000)
D = int(2)

# We have two radii
# Inner radius of 5
R_inner = int(5)
# Outer radius of 10
R_outer = int(10)

# Set a uniformly distributed variable for half the data that depends on the
# inner radius.  It is spread around 5
R1 = np.random.randn(int(N/2)) + R_inner
# Generate some angles, polar cordinates, that are uniformly distributed
theta = 2*np.pi*np.random.random(int(N/2))
# Convert the polar coordinates into (x,y) coordinates.
# We want to transpose that so N goes along the rows
X_inner = np.concatenate([[R1*np.cos(theta)], [R1 * np.sin(theta)]]).T

# We are then going to do the same for the outer radius
R2 = np.random.randn(int(N/2)) + R_outer
theta = 2*np.pi*np.random.random(int(N/2))
X_outer = np.concatenate([[R2*np.cos(theta)], [R2 * np.sin(theta)]]).T

X = np.concatenate([X_inner, X_outer])
# Set our targets.  First half to 0 and second half to 1 although order
# Does not matter.
T = np.array([0]*int(N/2) + [1]*int(N/2))

plt.scatter(X[:,0], X[:,1], c=T)
plt.show()


# add a column of ones for the bias term
ones = np.ones((N,1))

# The trick with the donut problem is to create another column that represents
# the radius of the plan
r = np.zeros((N,1))
for i in range(0,N):
    r[i] = np.sqrt(X[i,:].dot(X[i,:]))

Xb = np.concatenate((ones, r, X), axis = 1)

# randomly initialize weights
w = np.random.rand(D+2)

# Calculate model output
z = Xb.dot(w)

def sigmoid(z):
    return 1 / (1+np.exp(-z))

Y = sigmoid(z)

# calculate cross-entropy error
def cross_entropy(T, Y):
    E = 0
    for i in range(0,N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E

learning_rate = 0.0001
error = []
for i in range(0,5000):
    e = cross_entropy(T,Y)
    error.append(e)
    if i % 100 == 0:
        print(e)

    w += learning_rate * ( Xb.T.dot(T-Y) - 0.01*w)

    Y = sigmoid(Xb.dot(w))

plt.plot(error)
plt.title("Cross-entropy")
plt.show()

plt.plot(Y)
plt.show()

print("Final w:",w)
print("Final classification rate:", 1 - np.abs(T - np.round(Y)).sum() / N)
