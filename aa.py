import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import time

import warnings

warnings.filterwarnings(action="ignore")

pd.set_option('display.width', 1000)
pd.set_option('display.max_column', 35)
print("\n")
print("ML solution proposed by : Vivek Kumar Singh")
print("Email Id : viveksingh84527@gmail.com")
print("Student Id : 371926")

data = pd.read_csv('HeartAttack_data.csv')
print("\n")
print(data.dtypes)
print("\n")
print("\nSample HeartAttack dataset head(5) :- \n", data.head(5))

print("\n\n\nShape of the HeartAttack dataset  data.shape = ", end="")
print(data.shape)

print("\n\n\nHeartAttack  data decription : \n")
print(data.describe())

print("\n\n\ndata.num.unique() : ", data.num.unique())

list_drop = ['slope', 'ca', 'thal']
data.drop(list_drop, axis=1, inplace=True)
data.replace('?', np.nan, inplace=True)
print("\nAfter replacing the ? symbol by integer value. Update columns are :-\n", data)

print(data.isnull().sum())
column = ['trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang']
data[column] = data[column].fillna(data.mode().iloc[0])
data['age'].fillna(data['age'].mean(), inplace=True)
data['sex'].fillna(data['sex'].mean(), inplace=True)
data['cp'].fillna(data['cp'].mean(), inplace=True)
data['oldpeak'].fillna(data['oldpeak'].mean(), inplace=True)
print(data)

data.loc[:, 'trestbps':'exang'] = data.loc[:, 'trestbps':'exang'].applymap(float)
data.astype('float64')

print(data.info())

print("\ndata.num.unique() : ", data.num.unique())

print("\ndata.groupby('num').size()\n")
print(data.groupby('num').size())

plt.hist(data['num'])
plt.title(('num (1=Yes , 0=No)'))
plt.show()


data.plot(kind='density', subplots=True, layout=(3, 4), sharex=False, fontsize=1)
plt.show()

names = ['age', 'sex', ' cp', ' trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'num']
fig = plt.figure()
ax1 = fig.add_subplot(111)
cax = ax1.imshow(data.corr())
ax1.grid(True)
plt.title('Heart Attack Attributes Correlation')
# Add colorbar, make sure to specify tick locations to match desired ticklabels
ax1.set_xticklabels(names)
ax1.set_yticklabels(names)
fig.colorbar(cax, ticks=[.75, .8, .85, .90, .95, 1])

plt.show()

# Finally, we'll split the data into predictor variables and target variable,


Y = data['num'].values
X = data.drop('num', axis=1).values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=21)

models_list = []
models_list.append(('CART', DecisionTreeClassifier()))
models_list.append(('SVM', SVC()))
models_list.append(('NB', GaussianNB()))
models_list.append(('KNN', KNeighborsClassifier()))

num_folds = 10

results = []
names = []

for name, model in models_list:
    kfold = KFold(n_splits=num_folds, random_state=21)
    startTime = time.time()
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    endTime = time.time()
    results.append(cv_results)
    names.append(name)
    print("%s: %f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), endTime - startTime))

# Performance Comparision
# ------------------------------
fig = plt.figure()
fig.suptitle('Performance Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Standardize the dataset
pipelines = []

pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()), ('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()), ('SVM', SVC())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()), ('NB', GaussianNB())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsClassifier())])))

results = []
names = []

print("\n\n\nAccuracies of algorithm after scaled dataset\n")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    kfold = KFold(n_splits=num_folds, random_state=123)
    for name, model in pipelines:
        start = time.time()
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        end = time.time()
        results.append(cv_results)
        names.append(name)
        print("%s: %f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), end - start))
# Performance Comparison after Scaled Data
# ----------------------------------------

fig = plt.figure()
fig.suptitle('Performance Comparison after Scaled Data')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Application of SVC on dataset
# Let's fit the SVM to the dataset and see how it performs given the test data.

# prepare the model

scaler = StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)
model = SVC()
start = time.time()
model.fit(X_train_scaled, Y_train)  # Training of algorithm
end = time.time()
print("\n\nSVM Training Completed. It's Run Time: %f" % (end - start))

# estimate accuracy on test dataset
X_test_scaled = scaler.transform(X_test)
predictions = model.predict(X_test_scaled)
print("All predictions done successfully by SVM Machine Learning Algorithms")
print("\n\nAccuracy score %f" % accuracy_score(Y_test, predictions))

print("\n\n")
print("confusion_matrix = \n")
print(confusion_matrix(Y_test, predictions))

from sklearn.externals import joblib

filename = "finalized_HeartAttak_model.sav"
joblib.dump(model, filename)
print("Best Performing Model dumped successfully into a file by Joblib")
print("\n")
print("\n")

print("ML solution proposed by : Vivek Kumar Singh")
print("Email Id : viveksingh84527@gmail.com")
print("Student Id : 371926")
