import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('Hitters.csv')
d0 = df.dropna()
y = d0.Salary
x0 = d0.drop(['Salary'],axis=1)
x2 = x0.loc[:,x0.dtypes == object]
x = pd.get_dummies(x0,columns = ['League','Division','NewLeague'],
 drop_first=True)
X = x.astype('float64')
X_train,X_test,y_train,y_test = train_test_split(X,y,
 test_size=0.5,
 random_state=0)
alphas = 10**np.linspace(10,-2,100)
model = Ridge(normalize = True)
mspes = []
for i in alphas:
 model.set_params(alpha = i)
 model.fit(X_train, y_train)
 test_mspe = mean_squared_error(y_test, model.predict(X_test))
 mspes.append(test_mspe)
df_mspe = pd.DataFrame({'mspes': mspes})
df_mspe.index = alphas
df_mspe.index.name = 'alpha'
df_mspe.plot(figsize=(15, 8), grid=True, logx=True, xlim=(10 ** (-3), 10 ** 12))
mspes.index(min(mspes))
print(alphas[99])