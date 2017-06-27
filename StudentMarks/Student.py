## pridicting student marks depend upon time they
#put in study

import quandl
import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression

df = pd.read_csv('C:\\Users\\Abhishek\\Desktop\\MachineLearning\\S\\data.csv')

df.fillna(-99999,inplace = True)

X = np.array(df['hour'])
y = np.array(df['marks'])
X = X.reshape(-1,1)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size = 0.2)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)
print(accuracy)