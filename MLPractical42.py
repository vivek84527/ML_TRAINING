from sklearn.preprocessing import StandardScaler
from pandas import read_csv
from numpy import set_printoptions
import numpy as np

filename = 'indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test',
         'mass', 'pedi', 'age', 'class']

dataframe = read_csv(filename, names=names)
array = dataframe.values

# separate array into input and output components
X = array[:, 0:8]
Y = array[:, 8]

scaler = StandardScaler()
rescaledX = scaler.fit_transform(X)

print(rescaledX[:30, :])

print("\n\nMean of First coloum=")
print(np.mean(rescaledX[:, 0]))
