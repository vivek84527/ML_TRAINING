# Feature Selection : 4 Ways
# -------------------------------------
#   1. Univariate Feature Selection.
#   2. Recursive Feature Elimination.(RFE)
#   3. Principle Component Analysis(PCA).
#   4. Feature Importance Selection.


import pandas as pd
from numpy import set_printoptions

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import warnings
warnings.filterwarnings("ignore")

# load data
filename = 'indians-diabetes.data.csv'
hnames = ['preg', 'plas', 'pres', 'skin', 'test',
         'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=hnames)

array = dataframe.values

X = array[ : , 0:8]
Y = array[:,8]

# feature extraction
test = SelectKBest(score_func=chi2, k=3)
fit = test.fit(X, Y)
# summarize scores
set_printoptions(precision=3)
print(  fit.scores_   )
features = fit.transform(X)

# summarize selected features
print( "\n\n"  )
print(features[0:20,:])
