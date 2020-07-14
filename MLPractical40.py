# Data Transformation :

#  Prepare the Data For Machine Learning
# ***************************************

# 4 Ways to Prepare the Data For Machine Learning
# -------------------------------------------------
# 1. Rescale data.   (custom range)
# 2. Standardize data.
# 3. Normalize data.  ( 0 to 1)
# 4. Binarize data.
# Steps of Data Transforms
# ------------------------
# Step-1: Load the dataset from a URL.

# Step-2: Split the dataset into the input and
#        output variables for machine learning.

# Step-3: Apply a pre-processing transformation
#      technique to transform the input variables.

# Step-4: Summarize the data to show the change.


import pandas as pd
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler

filename = 'indians-diabetes.data.csv'
hnames = ['preg', 'plas', 'pres', 'skin', 'test',
          'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=hnames)

array = dataframe.values
# separate array into input and output components
X = array[:, 0:8]  # [ row , cols ]
Y = array[:, 8]
scaler = MinMaxScaler(feature_range=(1, 10))  # Range

# First Method
rescaledX = scaler.fit_transform(X)

# summarize transformed data
set_printoptions(precision=2)

print(rescaledX[0:30, :])  # [ row , cols ]
