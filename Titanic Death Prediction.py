
# coding: utf-8

# In[ ]:

#PART-1
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
get_ipython().magic('matplotlib inline')

#PART-2
#load the train and test data
train=pd.read_csv('titanic_train.csv')
test=pd.read_csv('titanic_test.csv')

#see the shape an dinformation of both train and test and identify any missing values
train.shape
train.info()
print('------------------------------')
print('The Number of Empty Rows/Values in Each Feature:')
train.isnull().sum()
test.info()
print('------------------------------')
print('The Number of Empty Rows/Values in Each Feature:')
test.isnull().sum()

#PART-3
#>First find out the outliers and drop them,as they distrub the entire data but at sometimes we have to look at the outliers which may be apprpopriate
#>Next find out the missing values and fill them out using feature engineering and Re-Process the data


#Here we detected the outliers using 'TUKEY METHOD',which is based upon the 'Inter Quartile Range'
def detect_outliers(df,n,features):
    outlier_indices=[]
    for col in features:
        Q1=np.percentile(df[col],25)
        Q3=np.percentile(df[col],75)
        IQR = Q3-Q1
        outlier_step=1*IQR
        oulier_list_col=df[(df[col]<Q1-outlier_step)|(df[col]>outlier_step+Q3)].index
        outlier_indices.extend( oulier_list_col)
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    return multiple_outliers    

#I'm looking out for the outliers in the below four features
outliers_drop=detect_outliers(train,2,["Age","SibSp","Parch","Fare"])
train.loc[outliers_drop]

#Drop the outliers
train=train.drop(outliers_drop,axis=0).reset_index(drop=True)

#merge/add both the data so that we can combinely do the Feature engineering for both the sets
data_full=pd.concat([train,test],axis=0).reset_index(drop=True)
data_full=data_full.fillna(np.nan)
data_full.isnull().sum()

#Look out for the correlation matrix and see how the features are correlated with the Survival
train.corr()

#check for the nul vallues in the fare column and fill it with median as there is only as single NAN value 
#And since the fare is very skewed,which will be a problematic in the prediction time so we make sure it is not skewed by using 'log'
data_full.Fare.isnull().sum()
data_full.Fare=data_full.Fare.fillna(data_full.Fare.median())
data_full.Fare=data_full.Fare.map(lambda x:np.log(x) if x>0 else 0)

#Look out for the features that how it is affecting the survival
sns.factorplot(x='SibSp',y='Survived',data=train,kind='bar')
sns.factorplot(x='Parch',y='Survived',data=train,kind='bar')
sns.FacetGrid(train,col='Survived').map(sns.distplot, "Age")
sns.factorplot(x='Pclass',y='Survived',data=train,kind='bar')
sns.factorplot(x='Pclass',y='Survived',hue='Sex',data=train)

#checking for the null values in the Embarked column and replacing NAN value with 'C' because it the most occuring value in that column
sns.factorplot(x='Embarked',y='Survived',data=train)
data_full.Embarked.isnull().sum()
data_full['Embarked']=data_full['Embarked'].fillna('C')
#replacing the string with numerical values
data_full['Embarked']=data_full['Embarked'].map({'S':0,'C':1,'Q':2})


#Now looking for the Age column,as there are many unknown values
#they are filled by using the same data of the related features
index_nan=list(data_full.Age[data_full.Age.isnull()].index)
for i in index_nan :
    age_med = data_full["Age"].median()
    age_pred = data_full["Age"][((data_full['SibSp'] == data_full.iloc[i]["SibSp"]) & (data_full['Parch'] == data_full.iloc[i]["Parch"]) & (data_full['Pclass'] == data_full.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred) :
        data_full['Age'].iloc[i] = age_pred
    else :
        data_full['Age'].iloc[i] = age_med

        
#PART-4
#Combinig the data and dividing them into categories
#Here I added the Family column from siblings and the parents including individual and dividing them into categories 
#i.e small families,medium families and large families
data_full['Family']=data_full['SibSp']+data_full['Parch']+1   
data_full['S_family']=data_full['Family'].map(lambda x: 1 if x<=2 else 0)
data_full['M_family']=data_full['Family'].map(lambda x: 1 if 2<x<=4 else 0)
data_full['L_family']=data_full['Family'].map(lambda x: 1 if x>4 else 0)

#Creating the dummy columns for the Pclass and we can do for all the categorical ones
data_full["Pclass"] = data_full["Pclass"].astype("category")
data_full = pd.get_dummies(data_full, columns = ["Pclass"],prefix="Pc")
data_full.columns

#replacing male and female with the numerical ones i.e 0 and 1
data_full['Sex']=data_full['Sex'].map({'male':0,'female':1})

#Drop the unnecessary columns ,But don't think them as unnecessary they might be useful in prediction 
data_full=data_full.drop('Cabin',axis=1)
data_full=data_full.drop('Name',axis=1)
data_full=data_full.drop('PassengerId',axis=1)
data_full=data_full.drop('Ticket',axis=1)



#PART-5
#Modelling of the data
len=train.shape[0]

#Now dividng both the test and train data using the length method and drop the survival form the test and divide the features and survival from the train data 
train=data_full[:len]
test=data_full[len:]
test.drop(labels=["Survived"],axis = 1,inplace=True)
features=train.drop(labels = ['Survived'],axis = 1)
survived=train['Survived']

#Now training of the data import train_test_split
from sklearn.cross_validation import train_test_split
features_train,features_test,survived_train,survived_test=train_test_split(features,survived,test_size=0.4,random_state=5)

#Now using svm function predict the deaths 
from sklearn import svm
clf=svm.SVC()
clf.fit(features_train,survived_train)
clf.score(features_test,survived_test)

#using Logistic regression 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg=LogisticRegression()
logreg.fit(features_train,survived_train)
predict=logreg.predict(features_test)
metrics.accuracy_score(predict,survived_test)


#using K-Nearest neighbors algorithm
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(features_train,survived_train)
predict=knn.predict(features_test)
metrics.accuracy_score(predict,survived_test)
prediction_svrvival_1=knn.predict(test)


#now checkin for the value of 'k' in k-nearest neighbors method and it id found to be 7
k_range=range(1,30)
accuracy=[]
for k in k_range:
    knn.fit(features_train,survived_train)
    predict=knn.predict(features_test)
    accuracy.append(metrics.accuracy_score(predict,survived_test))
max(accuracy)

#Instead of using the train_test_split we can use the K-Fold method for training then data
from sklearn.cross_validation import cross_val_score
#here we use the knn,where we get the accuracy than above
knn=KNeighborsClassifier(n_neighbors=7)
scores=cross_val_score(knn,features,survived,cv=10,scoring='accuracy')
scores.mean()

#After checking all the possibilities i got the better accuracy rate with the 'SVM Algorithm' and then save the result
predction_survival=clf.predict(test)
np.savetxt('digit.csv',predction_survival,delimiter=',')

