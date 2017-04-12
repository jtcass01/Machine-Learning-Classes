# need to sudo pip install xlrd to use pd.read_excel
# data is from:
# http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/mlr02.html

# The data (X1, X2, X3_ are for each patient.
# X1 = systolic blood pressure = Output
# X2 = age in years = Input
# X3 = weight in pounds = Input

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_excel('mlr02.xls')
X = df.as_matrix()

# Show age v blood pressure
plt.scatter(X[:,1], X[:,0])
plt.show()

# show weight v blood pressure
plt.scatter(X[:,2], X[:,0])
plt.show()

df['ones'] = 1

Y = df['X1']
X = df[['X2', 'X3', 'ones']]

X2only = df[['X2', 'ones']]
X3only = df[['X3', 'ones']]

def get_r2(X, Y):
    w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
    Yhat = X.dot(w)

    d1 = Y - Yhat
    d2 = Y - Y.mean()
    r2 = 1 - d1.dot(d1)/d2.dot(d2)
    return r2

print("r-squared using X2 and X3", get_r2(X,Y))
print("r-squared using X2", get_r2(X2only,Y))
print("r-squared using X3", get_r2(X3only,Y))
