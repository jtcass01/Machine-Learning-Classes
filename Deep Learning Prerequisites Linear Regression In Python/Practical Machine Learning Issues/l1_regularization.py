import numpy as np
import matplotlib.pyplot as plt

# number of data points
N = 50
# deminsionality
D = 50

#X matrix, uniformally distributed points of size NxD centered at 0 from -5 to +5
X = (np.random.random((N,D))-0.5)*10

print(X)

# All points but first three do not influence the output.
true_w = np.array([1,0.5,-0.5] + [0]*(D-3))

# Use gaussian random noise
Y = X.dot(true_w) + np.random.randn(N)*0.5

# Now we do gradient decent
costs =[]

w = np.random.randn(D) / np.sqrt(D)
learning_rate = 0.001
#l1 regularization term
l1 = 10.0
for t in range(0,500):
    Yhat = X.dot(w)
    delta = Yhat - Y
    w = w - learning_rate*(X.T.dot(delta) + l1*np.sign(w))

    mse = delta.dot(delta) / N
    costs.append(mse)

plt.plot(costs)
plt.show()

print("yhat", Yhat)
print("y", Y)
plt.plot(Yhat, label='yhat')
plt.plot(Y, label = 'y')
plt.legend()
plt.show()

print("final w:", w)

plt.plot(true_w, label='true w')
plt.plot(w, label='w_map')
plt.legend()
plt.show()
