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
