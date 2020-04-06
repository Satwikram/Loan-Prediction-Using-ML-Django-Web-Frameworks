# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 17:13:36 2019

@author: SATWIK RAM K
"""
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing Dataset
data = pd.read_csv('train.csv')

#Dropping Unwanted Information
data.drop('Loan_ID', axis = 1, inplace = True)

#Filling the nan values
data['Credit_History'].fillna(0, inplace = True)
data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0], inplace = True)
data['LoanAmount'].fillna(data['LoanAmount'].median(), inplace = True)
data['Self_Employed'].fillna('No', inplace = True )
data['Dependents'] = data['Dependents'].replace('3+', 4)
data['Dependents'].fillna(data['Dependents'].mode()[0], inplace = True)
data['Gender'].fillna(data['Gender'].mode()[0], inplace = True)
data['Married'].fillna('No', inplace = True)

#Getting Dummy Values
df = pd.get_dummies(data['Gender'], drop_first = True)
df1 = pd.get_dummies(data['Married'], drop_first = True)
df2 = pd.get_dummies(data['Education'], drop_first = True)
df3 = pd.get_dummies(data['Self_Employed'], drop_first = True)
df4 = pd.get_dummies(data['Property_Area'], drop_first = True)
df5 = pd.get_dummies(data['Loan_Status'], drop_first = True)

#Dropping originals
for i in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']:
    data.drop(i, axis = 1, inplace = True)
    
#Concatanating Dummies
data = pd.concat([data, df, df1, df2, df3, df4, df5], axis = 1)

#Taking X and Y
X = data.drop('Y', axis = 1)
y = data['Y']

#Standord Scalar
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
scaled_x = sc.fit_transform(X)

#Splitting the Data
from sklearn.model_selection import train_test_split, GridSearchCV
x_train, x_test, y_train, y_test = train_test_split(scaled_x, y, random_state = 42)

#Importing Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state = 0, n_estimators = 500, criterion = 'entropy',
                                    max_depth = 24, n_jobs = -1 )
classifier.fit(x_train, y_train)

#Predicting the data
y_pred = classifier.predict(x_test)

#Creating Confusion Matrix and Evaluating the accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test, y_pred)

#Importing Decision Tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(x_train,y_train)
clf.score(x_test,y_test)

# we are tuning three hyperparameters right now, we are passing the different values for both parameters
grid_param = {
    'criterion': ['gini', 'entropy'],
    'max_depth' : range(2,32,1),
    'min_samples_leaf' : range(1,10,1),
    'min_samples_split': range(2,10,1),
    'splitter' : ['best', 'random'],
    
}

grid_search = GridSearchCV(estimator=clf,
                     param_grid=grid_param,
                     cv = 10,
                   
                    n_jobs =-1)

grid_search.fit(x_train,y_train)

best_parameters = grid_search.best_params_
print(best_parameters)
print(grid_search.best_score_)

#Again training model
clf = DecisionTreeClassifier(criterion = 'entropy', max_depth =2, min_samples_leaf= 1, min_samples_split= 8, splitter ='random')
clf.fit(x_train,y_train)

y_pred1 = clf.predict(x_test)
cm1 = confusion_matrix(y_test, y_pred)
score1 = accuracy_score(y_test, y_pred)

# let's save the model
import pickle
# Writing different model files to file
with open( 'loan.sav', 'wb') as f:
    pickle.dump(clf,f)
    
with open('loanScaler.sav', 'wb') as f:
    pickle.dump(sc,f)
  
#Lets try with SVM
from sklearn.svm import SVC    
svm = SVC(kernel = 'linear', random_state = 0)
svm.fit(x_train, y_train)
    
#Predicting
y_pred2 = svm.predict(x_test)
cm2 = confusion_matrix(y_test, y_pred)

#Same Accuracy!
i





