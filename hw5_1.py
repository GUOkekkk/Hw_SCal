#1.读取数据=>2.创建sigmoid映射函数=>3.hΘ(x)函数=>4.最大似然函数=>5.梯度下降=>6.停止策略

#1.读取数据
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#读取数据
pdData=pd.read_csv('LogiReg_data.txt',header=None,names=['Exam 1','Exam 2','Admitted'])
#将数据分为X,y
pdData.insert(0,'Ones',1)
orig_data=pdData.values
cols=orig_data.shape[1]
X=orig_data[:,0:cols-1]
y=orig_data[:,cols-1:cols]
# print(X)

positive=pdData[pdData['Admitted']==1]
negative=pdData[pdData['Admitted']==0]

#2.创建sigmoid函数
def sigmoid(z):
    return 1/(1+np.exp(-z))

#3.创建hΘ(x)函数
def model(X,threta):
    sig=sigmoid(np.dot(X,threta.T))
    return sig

#4.求最大似然函数,即为最大似然函数的log形式
def cost(X,y,theta):
    #model即为概率映射函数
    temp=model(X,theta)
    left=np.multiply(y,np.log(temp))
    right=np.multiply(1-y,np.log(temp))
    return -(np.sum(left-right))/len(y)

#5.定义梯度下降函数，实现梯度下降
def gradient(X,y,theta):
    grad=np.zeros(theta.shape)
    error=(model(X,theta)-y).ravel()
    leng=len(threta[0])
    for i in range(leng):
        temp=np.multiply(error,X[:,i])
        grad[0,i]=sum(temp)/len(X)
    return grad

#6.设定不同的停止策略
#迭代次数，损失值，梯度
STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2

def stopCriterion(type,value,threshold):
    if type==0: return value>threshold
    elif type==1: return abs(value[-1]-value[-2])<threshold
    elif type==2: return np.linalg.norm(value)<threshold

#洗牌
def shuffleData(data):
    np.random.shuffle(data)
    cols=data.shape[1]
    X=data[:,0:cols-1]
    y=data[:,cols-1:]
    return X,y

import time
def descent(data,threta,batchSize,stopType,thresh,alpha):
    init_time=time.time()
    i=0
    k=0
    X,y=shuffleData(data)
    grad=np.zeros(threta.shape)
    costs=[cost(X,y,threta)]
    n=len(data)
    while True:
        grad=gradient(X[k:k+batchSize],y[k:k+batchSize],threta)
        k=batchSize+k
        if(k>=n):
            k=0
            X,y=shuffleData(data)
        threta=threta-alpha*grad
        costs.append(cost(X,y,threta))
        i=i+1

        if stopType == STOP_ITER:
            value = i
        elif stopType == STOP_COST:
            value = costs
        elif stopType == STOP_GRAD:
            value = grad
        if stopCriterion(stopType, value, thresh): break

    return threta,i-1,costs,grad,time.time()-init_time

def runExpe(data, theta, batchSize, stopType, thresh, alpha):
    #import pdb; pdb.set_trace();
    theta, iter, costs, grad, dur = descent(data, theta, batchSize, stopType, thresh, alpha)
    name = "Original" if (data[:,1]>2).sum() > 1 else "Scaled"
    name += " data - learning rate: {} - ".format(alpha)
    if batchSize==len(data): strDescType = "Gradient"
    elif batchSize==1:  strDescType = "Stochastic"
    else: strDescType = "Mini-batch ({})".format(batchSize)
    name += strDescType + " descent - Stop: "
    if stopType == STOP_ITER: strStop = "{} iterations".format(thresh)
    elif stopType == STOP_COST: strStop = "costs change < {}".format(thresh)
    else: strStop = "gradient norm < {}".format(thresh)
    name += strStop
    print ("***{}\nTheta: {} - Iter: {} - Last cost: {:03.2f} - Duration: {:03.2f}s".format(
        name, theta, iter, costs[-1], dur))
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(np.arange(len(costs)), costs, 'g')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(name.upper() + ' - Error vs. Iteration')
    return theta

threta=np.zeros([1,3])
runExpe(orig_data,threta,batchSize=60,stopType=STOP_ITER,thresh=10000,alpha=0.000001)
plt.show()
