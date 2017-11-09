import numpy as np

a = np.random.randn(5) #Creates 5 random gaussian variables

#Note: when printing a and a.T as row vectors, you get the same output!!!
print a
print a.shape
print a.T
print a.T.shape
print '\n'

# This gives back a number.
print np.dot(a,a.T)
print '\n'

#His suggestion is to not use data structures that use this (n,) shape
#instead...
a = np.random.randn(5,1) #This commits A to be a 5,1 column vector
print a
print a.shape
print a.T
print a.T.shape #note output differences such as the number of brackets.
# Bottom example uses two square brackets while the top example only uses 1
print '\n' 

# This gives you the outer product now.
print np.dot(a,a.T)
