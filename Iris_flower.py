import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV

df=pd.read_csv('Iris_flower.csv')
X=df[['SepalLengthCm','SepalWidthCm','PetalLengthCm',
      'PetalWidthCm']]
y=df['Species']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)


#** search parameters
param_grid={'C':[0.1,1,10,100],'gamma':[1,0.1,0.01,0.001]}
grid=GridSearchCV(SVC(),param_grid,verbose=2)
grid.fit(X_train,y_train)
print(grid.best_estimator_)


#***predition
predition=grid.predict(X_test)
c=confusion_matrix(y_test,predition)
c1=classification_report(y_test,predition)
print(c1)
