import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

data= pd.read_csv('College_Data.csv')
data.drop(data.columns[[0]],axis=1,inplace=True)

#***Kmean
kmeans=KMeans(n_clusters=2) #private and public school
kmeans.fit(data.drop('Private',axis=1))

#*** convert No & Yes variable into 0:public or 1:private
#**# * convert No & Yes variable into 0:public or 1:private
def convert(Private):
    if Private=='Yes':
        return 1
    else:
        return 0

data['cluster']=data['Private'].apply(convert)
#**evalute
c=confusion_matrix(data['cluster'],kmeans.labels_)
c1=classification_report(data['cluster'],kmeans.labels_)

print(c)
print(c1)







