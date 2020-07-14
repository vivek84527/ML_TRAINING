# 4) Feature Importance Selection
# --------------------------------------
# Its based on Gini Importance
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

# load data
filename = 'indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test',
         'mass', 'pedi', 'age', 'class']

dataframe = pd.read_csv(filename, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]

# feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
scores = model.feature_importances_
print(scores)
# 4) Feature Selection based on Gini Importance Index
# -----------------------------------------------------
# Its based on Gini Importance
