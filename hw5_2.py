import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error

# Import all the necessary packages

df = pd.read_csv('Hitters.csv')
# Import the data file Hitters.csv

df_0 = df.dropna()
# Remove all rows with missing values

train_df, test_df = train_test_split(df_0, test_size=0.5, random_state=0)
# Split the data set into a training and test set

y_train = train_df.Salary
X_train = pd.get_dummies(train_df.drop(['Salary'], axis=1),
                         columns=['League', 'Division', 'NewLeague'],
                         drop_first=True)
# Substitute categorical cols with dummy vars of train data
y_test = test_df.Salary
X_test = pd.get_dummies(test_df.drop(['Salary'], axis=1),
                        columns=['League', 'Division', 'NewLeague'],
                        drop_first=True)
# Substitute categorical cols with dummy vars of test

alphas = 10 ** np.linspace(-2, 10, 100)
# Create an array of 100 α-values
print(alphas)
model_1 = Ridge(normalize=True)
mspes = []
for i in alphas:
    model_1.set_params(alpha=i)
    model_1.fit(X_train, y_train)
    test_mspe = mean_squared_error(y_test, model_1.predict(X_test))
    mspes.append(test_mspe)
plt.plot(np.log10(alphas), mspes, label='the test_mspe of ridge regression')
plt.xlabel('alpha values on the (log10)')
plt.ylabel('the test_mspe values')
# Plot the test_mspe values (on y-axis) with alpha values on the (log10) x-axis.

df_mspe = pd.DataFrame(mspes, index=alphas, columns=['Mspes'])
alpha_min = df_mspe.idxmin()
#  Find the value of alpha minimizing the test_mspe

model_2 = LinearRegression().fit(X_train, y_train)
test_mspe_LR = mean_squared_error(y_test, model_2.predict(X_test))
# Fit a linear regression model and find the test_mspe.

plt.axhline(y=test_mspe_LR, color='red', linestyle='--',
            label='the test_mspe of liner regression')
plt.legend()
plt.show()
# Draw the picture

alpha_interval = df_mspe[df_mspe.Mspes < test_mspe_LR].index
# Identify the set of alpha values that result in a ridge regression model with smaller test_mspe than the linear regression.

print(alpha_min)
print(alpha_interval)
# print the result
