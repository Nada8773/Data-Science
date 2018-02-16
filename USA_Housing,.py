import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

# linear regression
df = pd.read_csv('USA_Housing.csv')
X=df[['Avg. Area Income', 'Avg. Area House Age',
       'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms',
       'Area Population']]
y=df['Price']
X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.4,random_state=101)
lm=LinearRegression()
lm.fit(X_train,y_train)
predict=lm.predict(X_test)
print(X_test)