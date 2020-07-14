#Dimension Reduction Method
#-----------------------------
import pandas as pd
from sklearn.decomposition import PCA

# load data
filename = 'indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test',
         'mass', 'pedi', 'age', 'class']

dataframe = pd.read_csv(filename, names=names)

array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

# feature extraction
pca = PCA(n_components=3)

fit = pca.fit(X)

resultX = pca.transform(X)
print( "\nResult : \n" , resultX  )

# summarize components
print(  "Explained Variance:" ,fit.explained_variance_ratio_  )
