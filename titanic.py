# logistic regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

train=pd.read_csv('titanic_train.csv')
#*** Analyize your data
#sns.heatmap(train.isnull()) # to show the missing data
#sns.countplot(x='Survived',hue='Sex',data=train)

#***fill the missing data

#*** using the average age for each pclass to fill the missing data
#sns.boxplot(x='Pclass',y='Age',data=train) # to get the average age for each class
#plt.show()
def miss_age(col):
    Age = col[0]
    Pclass=col[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age
train['Age']=train[['Age','Pclass']].apply(miss_age,axis=1) # 1=columns
#*** drop the cabin col as it has many missing data
train.drop('Cabin',axis=1,inplace=True)

train.dropna(inplace=True) # drop the row that have missing data
#*** convert variable into 0 or 1
embarked=pd.get_dummies(train['Embarked'],drop_first=True)
sex=pd.get_dummies(train['Sex'],drop_first=True)

#sns.heatmap(train.isnull())
#plt.show()

#*** put your new col into data
train=pd.concat([train,embarked,sex],axis=1)

#***Remove some columns from data
train.drop(['Name','PassengerId','Ticket','Sex','Embarked'],axis=1,inplace=True)

#print(train.head())

#*** split data
X=train.drop('Survived',axis=1) #feature
y=train['Survived'] #label
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
#***train model
lm=LogisticRegression()
lm.fit(X_train,y_train)

#***prediction
prdiction=lm.predict(X_test)

#***evaluate your model
evaluate=classification_report(y_test,prdiction)
#or by using confusion matrix
c=confusion_matrix(y_test,prdiction)
#print(c)

accuracy=lm.score(X_test,y_test)
print(accuracy)
#print(train.head())
#print(prdiction)
#print(evaluate)
