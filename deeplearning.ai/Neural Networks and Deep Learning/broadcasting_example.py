import numpy as np

# This is a matrix with columns: apples, beef, eggs, potatoes
#and rows: carbs, protein, fat 
A = np.array([[56.0,0.0,4.4,68.0],
              [1.2,104.0,52.0,8.0],
              [1.8,135.0,99.0,0.9]])

print(A, 'Initial matrix')
print('\n')

# Goal is to calculate the percent of valories from carbs, protein, and fat.
#To do this we sum the rows of each column and divdie each value in the column
#by that number*100
cal = A.sum(axis=0) # The axis parameter tells it to sum over the columns, not the rows
print(cal, 'total calories per column')
print('\n')
percentages = (A/cal.reshape(1,4))  *   100
print(percentages, 'caloric percentages with respect to column')
