import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Import all the necessary packages

df = pd.read_csv('dataset.csv')
# Import the data file dataset.csv
df.dropna()
x = df[['x1', 'x2']]
y = df['y']
train_x, test_x, train_y, test_y = train_test_split(x, y,
                                                    test_size=1 - 0.4,
                                                    random_state=0,
                                                    stratify=y)
# Split the data set into a training and test set

k_range = range(1, 20)
# set the range of the neighbors

k_scores_test = []
k_scores_train = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_x, train_y)
    k_scores_train.append(knn.score(train_x,train_y))
    k_scores_test.append(knn.score(test_x, test_y))
# Fit a KNN model to predict y

plt.plot(k_range, k_scores_test, label = 'test accuracy')
plt.plot(k_range, k_scores_train, label = 'train accuracy')
plt.xlabel('the number of neighbors')
plt.ylabel('the accuracy')
plt.legend()
# plt.show()
# use the plot to find the best number of neighbors

LR = LogisticRegression(solver='lbfgs')
LR.fit(train_x, train_y)
acc_LR = LR.score(test_x, test_y)
# Find test accuracy rate of LogisticRegression

train_df = train_x.copy()
train_df['y'] = train_y
train_df_1 = train_df[train_df.y == 1]
train_df_0 = train_df[train_df.y == 0]
plt.figure()
plt.scatter(x=train_df_1['x1'], y=train_df_1['x2'], color='r', label='y=1')
plt.scatter(x=train_df_0['x1'], y=train_df_0['x2'], label='y=0')
plt.legend()
#plt.show()
# Draw a scatterplot of X1 (x-axis) vs X2, (y-axis)

def Create_newdataset(df):
    df1 = df.copy()
    df1['x1^2'] = df['x1'] * df['x1']
    df1['x2^2'] = df['x2'] * df['x2']
    df1['x1*x2'] = df['x1'] * df['x2']
    return df1

train_x_new = Create_newdataset(train_x)
test_x_new = Create_newdataset(test_x)
# Create a new datafile with x1^2,x2^2 and x1*x2

LR_new = LogisticRegression(solver='lbfgs', max_iter=10000)
LR_new.fit(train_x_new, train_y)
acc_LR_new = LR_new.score(test_x_new, test_y)
print(acc_LR)
print(acc_LR_new)
#  Find the test accuracy rate.
