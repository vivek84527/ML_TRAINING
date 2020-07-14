# Binarization
from sklearn.preprocessing import Binarizer
import pandas as pd

filename = 'indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test',
         'mass', 'pedi', 'age', 'class']

dataframe = pd.read_csv(filename, names=names)
array = dataframe.values

# separate array into input and output components
X = array[:, 0:8]
Y = array[:, 8]

binarizer = Binarizer(threshold=5)

binaryX = binarizer.fit_transform(X)
# summarize transformed data
print(binaryX[0:30, :])
