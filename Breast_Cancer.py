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

#** dataSet
from sklearn.datasets import load_breast_cancer

#*** organize or data
data = load_breast_cancer()
df=pd.DataFrame(data['data'],columns=data['feature_names'])

#***split data
X=df
y=data['target']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=101)

#***SVM classifier
ml=SVC(C=1000 ,gamma=0.00001, kernel='rbf')
ml.fit(X_train,y_train)

#***predition
predition=ml.predict(X_test)
c=confusion_matrix(y_test,predition)
c1=classification_report(y_test,predition)

#** search parameters
#param_grid={'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
#grid=GridSearchCV(SVC(),param_grid,verbose=3)
#grid.fit(X_train,y_train)
#print(grid.best_estimator_) to get best parameter for svm

