# Normalize data
from sklearn.preprocessing import Normalizer
from pandas import read_csv
from numpy import set_printoptions

filename = 'indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test',
         'mass', 'pedi', 'age', 'class']

dataframe = read_csv(filename, names=names)
array = dataframe.values

# separate array into input and output components
X = array[:, 0:8]
Y = array[:, 8]

scaler = Normalizer()
normalizedX = scaler.fit_transform(X)

# summarize transformed data
set_printoptions(precision=2)
print(normalizedX[0:30, :])
