# Data Types for Each Attribute
import pandas

filename = "indians-diabetes.data.csv"
hnames = ['preg', 'plas', 'pres',
          'skin', 'test', 'mass',
          'pedi', 'age', 'class']

dataframe = pandas.read_csv(filename, names=hnames)

# Peek at Your Data
# review the first 10 rows of your data using the head()
#  function on the Pandas DataFrame.
print(dataframe.head(10))
print("-*-" * 20)
# Dimensions of Your Data
print("dataframe.shape : ", dataframe.shape)
print("-*-" * 20)
# Data Type For Each Attribute
print(dataframe.dtypes)
print("-#-" * 20)
# Descriptive Statistics
pandas.set_option('display.width', 1000)
pandas.set_option('precision', 2)

print("description = \n", dataframe.describe())

print("-*-" * 20)
# Class Distribution (Classification Only)
print()

class_counts = dataframe.groupby('class').size()

print(class_counts)
print("-#-" * 20)

# Correlations Between Attributes
# Correlation refers to the relationship between two variables
# and how they may or may not change together.
correlations = dataframe.corr(method='pearson')

print(correlations)
