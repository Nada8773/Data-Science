import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Ecommerce Customers.csv')
X=df[['Avg. Session Length','Time on App',
       'Time on Website', 'Length of Membership']]
y=df['Yearly Amount Spent']
X_train ,X_test,y_train ,y_test= train_test_split(X,y,test_size=0.3,random_state=101)
lm=LinearRegression()
lm.fit(X_train,y_train)
print(lm.coef_)
#plt.scatter(y_test,lm.predict(X_test))
#plt.show()