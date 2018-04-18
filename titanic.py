#!/usr/bin/python

#Data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

#visualizatioin

import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

#machine learning

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv("/home/tom/Documents/study/kaggle_competition/train.csv")
test = pd.read_csv("/home/tom/Documents/study/kaggle_competition/test.csv")
test1 = test
combine = [train,test]

print (train.columns.values)
print ("Train set\n",train.info())
print ("Test set\n",test.info())
print ("-----------------------------------------------")

print (train.describe())

print ("----------String------------")
#print (train.describe(include=['0']))
print "Pclass based survival stati\n"

print train[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by = 'Survived')

print "Sex based survival stati\n"
print train[['Sex','Survived']].groupby(['Sex'], as_index=False).mean()

print "Sibling based survival\n"
print train [['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by = 'Survived',ascending = False)

print "Parent based survival\n"
print train[['Parch','Survived']].groupby(['Parch'],as_index=False).mean().sort_values(by = 'Survived',ascending=False)









print "Comparing age with survived "

#g = sns.FacetGrid(train, col='Survived')
g = sns.FacetGrid(train,col='Survived')
g.map(plt.hist, 'Age', bins=100)
#plt.show()


print "Comapring Pclas and survived with age"

g=sns.FacetGrid(train,col='Survived', row='Pclass')
g.map(plt.hist,'Age',bins=20)
g.add_legend();
#plt.show()

print "adding sex to the plot"

grid = sns.FacetGrid(train,row='Embarked',size=2.2,aspect=1.6)
grid.map(sns.pointplot,'Pclass','Survived','Sex',palette='deep')
grid.add_legend()
#plt.show()

print "Fare vise checkig"

g = sns.FacetGrid(train,row='Embarked',col='Survived',size=2.2,aspect=1.6)
g.map(sns.barplot,'Sex','Fare',alpha=.5,ci=None)
g.add_legend()
#plt.show()

print "Data Wrangling"

print "Before",train.shape,test.shape,combine[0].shape,combine[1].shape
#Columns Tickect and Cabin have no relevence so removing that
train = train.drop(['Ticket','Cabin'],axis=1)
test = test.drop(['Ticket','Cabin'],axis=1)
combine = [train,test]
print "After",train.shape,test.shape,combine[0].shape,combine[1].shape
print train.columns.values

#create new title feature
#Innorder to drop Name and PassengerId
for dataset in combine:
	dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.',expand=False)

print pd.crosstab(train['Title'],train['Sex'])

#Replace many titles with more common or classify them as rare

for dataset in combine:
	dataset['Title'] = dataset['Title'].replace(['Lady','Countes','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
	dataset['Title'] = dataset['Title'].replace('Mlle','Miss')
	dataset['Title'] = dataset['Title'].replace('Ms','Miss')
	dataset['Title'] = dataset['Title'].replace('Mme','Mrs')

print train[['Title','Survived']].groupby(['Title'], as_index=False).mean()

#Converting titles into ordinal-int

title_mapping = {'Mr': 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
	dataset['Title'] = dataset['Title'].map(title_mapping)
	dataset['Title'] = dataset['Title'].fillna(0)

print train.head()

#Dropping Name and PassengerId

train = train.drop(['Name','PassengerId'], axis=1)
test = test.drop(['Name','PassengerId'],axis=1)
print train.shape,test.shape
print train

combine = [train,test]
#Mapping sex to in

for dataset in combine:
	dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

print train
#combine = [train,test]
#Co-relating age ,Gender, Pclass

g = sns.FacetGrid(train,row='Sex', col = 'Pclass', size=2.2, aspect=1.6)
g.map(plt.hist,'Age',alpha=.5,bins = 20)
g.add_legend()

#plt.show()


guess_ages=np.zeros((2,3))
print guess_ages
combine=[train,test]
for dataset in combine:
	for i in range(0,2):
		for j in range(0,3):
			guess_df= dataset[(dataset['Sex']==i) & (dataset['Pclass'] == j+1)]['Age'].dropna()

			age_guess = guess_df.median()
			guess_ages[i,j] = int ( age_guess/0.5 + 0.5) * 0.5
	for i in range(0,2):
		for j in range(0,3):
			dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age'] = guess_ages[i,j]
	dataset['Age']=dataset['Age'].astype(int)
print guess_ages
print train
#train['AgeBand'] = pd.cut(dataset['Age'],5)
#print train[['AgeBand','Survived']].groupby(['AgeBand'],as_index=False).mean().sort_values(by='AgeBand',ascending=True)
	
#print age_guess
#print guess_ages
#print '------------e123213-232321323213232---------------------'
#print combine[0].Age


for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)
train['AgeBand'] = pd.cut(train['Age'],5)
print train[['AgeBand','Survived']].groupby(['AgeBand'],as_index=False).mean()

#Converting age band to numeric values

for dataset in combine:
	dataset.loc[dataset['Age'] <=16,'Age'] = 0
	dataset.loc[(dataset['Age']>16) & (dataset['Age']<=32), 'Age'] = 1
	dataset.loc[(dataset['Age']>32) & (dataset['Age']<=48), 'Age'] = 2
	dataset.loc[(dataset['Age']>48) & (dataset['Age']<=64), 'Age'] = 3
	dataset.loc[(dataset['Age']>64),'Age']
print train.head()
print test.head()
print '-----------------------------after=-----------------------------`'
#After converting age drop ageband

train = train.drop(['AgeBand'],axis=1)
combine = [train,test]
print train.head()
print test.head()

#Combining Parch and SibSp and creating new FamilySize

for dataset in combine:
	dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

print train[['FamilySize','Survived']].groupby(['FamilySize'],as_index=False).mean().sort_values(by='Survived',ascending = False)

#	Creating another variable IsAlone 

for dataset in combine:
	dataset['IsAlone']=0
	dataset.loc[dataset['FamilySize'] == 1,'IsAlone'] = 1



print train[['IsAlone','Survived']].groupby(['IsAlone'],as_index=False).mean().sort_values(by='Survived',ascending = False)

#After geting IsAlone variable drop Parch,SibSp,FamilySize


train = train.drop(['Parch','SibSp','FamilySize'],axis=1)
test = test.drop(['Parch','SibSp','FamilySize'],axis=1)
combine = [train,test]

#Creating artificial feature combining  Pclass and Age

for dataset in combine:
	dataset['Age*Class']= dataset.Age * dataset.Pclass

print train.loc[:,['Age*Class','Age','Pclass']]

#Since two missing values for Embarked values , finding the most common occurance of embarked


freq_port = train.Embarked.dropna().mode()[0]

print freq_port
#print train.head()

for dataset in combine:
	dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

#Analysing Embarked with survived


print train[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean().sort_values(by = 'Survived',ascending=False)

#Converting Embarked to numeric


for dataset in combine:
	dataset['Embarked'] =  dataset['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)

#Replacing missing fare values with the mean fare

test['Fare'].fillna(test['Fare'].dropna().median(),inplace=True)

print test.head()

#Creating fare band

train['FareBand'] = pd.qcut(train['Fare'],4)
print train[['FareBand','Survived']].groupby(['FareBand'],as_index=False).mean().sort_values(by='Survived',ascending=False)

#Coverting Fare to numeric values based on fareband

for dataset in combine:
	dataset.loc[(dataset['Fare']<=7.91),'Fare'] = 0
	dataset.loc[(dataset['Fare']>7.92) & (dataset['Fare'] <= 14.454),'Fare']= 1
	dataset.loc[(dataset['Fare']>14.454) & (dataset['Fare']<= 31),'Fare'] = 2
	dataset.loc[(dataset['Fare']>31),'Fare'] = 3
	dataset['Fare'] = dataset['Fare'].astype(int)
train = train.drop(['FareBand'],axis=1)

combine=[test,train]

#Now creating prediction Model 

X_train = train.drop('Survived',axis=1)
Y_train = train['Survived']
X_test = test

print "-----Xtrain--------------\n",X_train.head()
print "------Ytrain=========\n",Y_train.head()
print "-------Xtest-------------\n",X_test.head()

print X_train.shape,Y_train.shape,X_test.shape

#Logistic Regression

logreg = LogisticRegression()
random = RandomForestClassifier()

logreg.fit(X_train,Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print acc_log

#RandomForestClassifier
print "Random Forest "

random = RandomForestClassifier(n_estimators=100)
random.fit(X_train,Y_train)
Y_pred = random.predict(X_test)
acc_log = round(random.score(X_train, Y_train) * 100, 2)
print acc_log

#Decision Tree
print "Decision Tree"
tree = DecisionTreeClassifier()
tree.fit(X_train,Y_train)
Y_pred=tree.predict(X_test)
acc_tree = round(tree.score(X_train,Y_train) * 100, 2)
print acc_tree

# Creating Submission file

submission = pd.DataFrame({
	"PassengerId": test1['PassengerId'],
	"Survived": Y_pred})
submission.to_csv('/home/tom/Documents/study/kaggle_competition/submission.csv', index = False)
#print test.head()

