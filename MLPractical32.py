import warnings
warnings.filterwarnings(action="ignore")

import pandas
import urllib.request

hnames = ['sepal-length', 'sepal-width',
          'petal-length', 'petal-width',
          'class']

web_path = urllib.request.urlopen( "https://goo.gl/QnHW4g")

dataframe = pandas.read_csv(web_path,  names=hnames)

print  ( dataframe.shape  )

print( dataframe  )

print("\n\n\n")
print(dataframe.values)

print(dataframe.columns)

print("\n\n\n")
print(dataframe.dtypes)


print("\n\n\n")
print(dataframe.describe() )

print("\n\n\n")
print(dataframe.describe(include="all") )
