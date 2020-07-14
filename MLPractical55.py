import warnings

warnings.filterwarnings(action="ignore")

import pandas as pd

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

filename = 'indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass',
         'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]

test_data_size = 0.33
no_of_repetiions = 4

shufflesplit = ShuffleSplit(n_splits=no_of_repetiions,
                            test_size=test_data_size)

model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=shufflesplit)
print(results)

print("Accuracy: %.3f " % (results.mean() * 100.0))

print("Std.Deviation= %.3f" % (results.std() * 100.0))
