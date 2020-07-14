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




#Exploratory analysis
#Load the dataset and do some quick exploratory analysis.

data = pd.read_csv('BrainTumorData.csv', index_col=False)
print("\n\n\nSample BrainTumor dataset head(5) :- \n\n", data.head(5) )



print("\n\n\nShape of the BrainTumor dataset  data.shape = ", end="")
print( data.shape)
#(569, 33)


print("\n\n\nBrainTumor data decription : \n")
print( data.describe() )



#Data visualisation and pre-processing


#First thing to do is to enumerate the diagnosis column such that M = 1, B = 0.
#  Then, I set the ID column to be the index of the dataframe.
# Afterall, the ID column will not be used for machine learning


print( "\n\n\ndata.diagnosis.unique() : " , data.diagnosis.unique() )




#Replace M = 1   and B = 0
#Firts Trick:-
data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})

#Second Trick:-
#data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
print("\n\n\nAfter updation of  diagnosis feature: \n", data.head() )




plt.hist(data['diagnosis'])
plt.title('Diagnosis (M=1 , B=0)')
plt.show()



data = data.set_index('id')
print("\n\n\nAfter id feature is set as row index: \n", data)


del data['Unnamed: 32']
print("\n\nAfter Deletion of 'Unnamed: 32' column\n", data)




#Let's take a look at the number of Benign and Maglinant cases from the dataset.
# From the output shown below, majority of the cases are benign (0).

print("\n\n\ndata.groupby('diagnosis').size()\n")
print(data.groupby('diagnosis').size())

#diagnosis
#0    357
#1    212
#dtype: int64




#Next, we visualise the data using density plots to get a sense of the data distribution.
# From the outputs below, you can see the data shows a general gaussian distribution.
data.plot(kind='density', subplots=True, layout=(5,7), sharex=False, legend=False, fontsize=1)
plt.show()





#from matplotlib import cm as cm

fig = plt.figure()
ax1 = fig.add_subplot(111)
cax = ax1.imshow(data.corr() )
ax1.grid(True)
plt.title('Cancer Attributes Correlation')
# Add colorbar, make sure to specify tick locations to match desired ticklabels
fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
plt.show()




#Finally, we'll split the data into predictor variables and target variable,
# following by breaking them into train and test sets. We will use 20% of the data as test set.

Y = data['diagnosis'].values
X = data.drop('diagnosis', axis=1).values

X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.33, random_state=21)


#Baseline algorithm checking
#From the dataset, we will analysis and build a model to predict if a given set of
# symptoms lead to a cancerous BrainTumor.
# This is a binary classification problem, and a few algorithms are appropriate for use.
# Since we do not know which one will perform the best at the point,
# we will do a quick test on the few algorithms to get an early indication of how each of them perform.
# We will use K-Fold cross validation for each testing.

#The following  algorithms will be used,

#1) Classification and Regression Trees (CART),
#2) Support Vector Machines (SVM),
#3) Gaussian Naive Bayes (NB)
#4) k-Nearest Neighbors (KNN).




models_list = []
models_list.append(('CART', DecisionTreeClassifier()))
models_list.append(('SVM', SVC()))
models_list.append(('NB', GaussianNB()))
models_list.append(('KNN', KNeighborsClassifier()))


num_folds = 10

results = []
names = []

for name, model in models_list:
    kfold = KFold(n_splits=num_folds, random_state=123)
    startTime = time.time()
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    endTime = time.time()
    results.append(cv_results)
    names.append(name)
    print( "%s: %f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), endTime-startTime))



#CART: 0.912029 (0.039630) (run time: 0.138211)
#SVM: 0.619614 (0.082882) (run time: 0.164310)
#NB: 0.940773 (0.033921) (run time: 0.019228)
#KNN: 0.927729 (0.055250) (run time: 0.027202)




#Performance Comparision
#------------------------------
fig = plt.figure()
fig.suptitle('Performance Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()





#From the initial run, it looks like GaussianNB, KNN and CART performed the best
# given the dataset (all above 92% mean accuracy).
# Support Vector Machine has a surprisingly bad performance here.
# However, if we standardise the input dataset, it's performance should improve.





#Evaluation of algorithm on Standardised Data
#The performance of the machine learning algorithm could be improved if a
# standardised dataset is being used.

# The improvement is likely for all the models.

# I will use pipelines that standardize the data and build the model for each
#  fold in the cross-validation test harness.

# That way we can get a fair estimation of how each model with standardized data might perform on unseen data.






# Standardize the dataset
pipelines = []

pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC( ))])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))


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
        print( "%s: %f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), end-start))




#ScaledCART: 0.920966 (0.038259) (run time: 0.098808)
#ScaledSVM: 0.964879 (0.038621) (run time: 0.073377)
#ScaledNB: 0.931932 (0.038625) (run time: 0.027154)
#ScaledKNN: 0.958357 (0.038595) (run time: 0.040088)

#Notice the drastic improvement of SVM after using scaled data.


#Performance Comparison after Scaled Data
#----------------------------------------

fig = plt.figure()
fig.suptitle('Performance Comparison after Scaled Data')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()





#Application of SVC on dataset
#Let's fit the SVM to the dataset and see how it performs given the test data.

# prepare the model

scaler = StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)
model = SVC()
start = time.time()
model.fit(X_train_scaled, Y_train)   #Training of algorithm
end = time.time()
print( "\n\nSVM Training Completed. It's Run Time: %f" % (end-start))

#Run Time: 0.004889


# estimate accuracy on test dataset
X_test_scaled = scaler.transform(X_test)
predictions = model.predict(X_test_scaled)
print("All predictions done successfully by SVM Machine Learning Algorithms")
print("\n\nAccuracy score %f" % accuracy_score(Y_test, predictions))

print("\n\n")
print("confusion_matrix = \n")
print( confusion_matrix(Y_test, predictions))




from sklearn.externals import joblib
filename =  "finalized_BrainTumor_model.sav"
joblib.dump(model, filename)
print( "Best Performing Model dumped successfully into a file by Joblib")


