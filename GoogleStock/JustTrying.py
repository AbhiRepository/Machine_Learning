import quandl, math, datetime
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = quandl.get("WIKI/GOOGL")

df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)
forcast_out = int(math.ceil(0.01*len(df)))

df['lable'] = df[forecast_col].shift(-forcast_out)
X = np.array(df.drop(['lable'],1))
X = preprocessing.scale(X)
X_lately = X[-forcast_out:]
X = X[:-forcast_out]



df.dropna(inplace=True)
y = np.array(df['lable'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)

forcast_set = clf.predict(X_lately)

print(forcast_set, accuracy, forcast_out)

df['Forcast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forcast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix+=one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['Forcast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
