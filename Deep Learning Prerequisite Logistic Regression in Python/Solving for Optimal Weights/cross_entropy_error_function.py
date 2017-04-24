import numpy as np

N = 100
D = 2

X = np.random.randn(N,D)

# Set the first 50 points to be centered at X = -2 and Y = -2
# This is done by creating a matrix of 1s and multipying by -2
X[:50, :] = X[:50,:] - 2*np.ones((50,D))

#Now I can do the same thing for the other class, except I am going to center
# That data at x=2, y=2
X[50:, :] = X[50:, :] + np.ones((50,D))

#Now I am going to set the first 50 to zero and the second 50 to 1
T = np.array([0]*50 + [1]*50)
print("T",T)

#Now I am going to concatenate a column of 1s to the initial array
ones = np.array([[1]*N]).T
Xb = np.concatenate((ones,X), axis=1)

# randomly initialize th weights
w = np.random.randn(D+1)

# calculate the model output
z = Xb.dot(w)
print("z",z)

def sigmoid(z):
    return 1/(1+np.exp(-z))

Y = sigmoid(z)
print("Y",Y)

def cross_entropy(T, Y):
    E = 0
    for i in range(0, N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1-Y[i])
    return E

print("cross_entropy", cross_entropy(T, Y))

#This will work because we have equal variances in each classes.  Therefore:
#The weight only depends on the means
#Using equation: w.T = (u1.T - u2.T)*(inverse covariance matrix)
w = np.array([0,4,4])
z = Xb.dot(w)
Y = sigmoid(z)

print("z (closed form)",z)
print("Y (closed form)",Y)
print("cross_entropy (closed form solution)", cross_entropy(T, Y))
