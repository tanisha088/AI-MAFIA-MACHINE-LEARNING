#!/usr/bin/env python
# coding: utf-8

# In[20]:



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
  
## Data Preparation "
  
dfx = pd.read_csv('C:/Users/user/Desktop/PYCHARM_PROJECTS/Logistic_X_Train.csv')
dfy = pd.read_csv('C:/Users/user/Desktop/PYCHARM_PROJECTS/Logistic_Y_Train.csv')
dftest = pd.read_csv('C:/Users/user/Desktop/PYCHARM_PROJECTS/Logistic_X_Test.csv')

x = dfx.values
y = dfy.values
xtest = dftest.values
x=x[:,0]
print(x)
print(y)
plt.scatter(x,y)
X = (x-x.mean())/x.std()
Y = y
plt.scatter(X,Y)
plt.show()
## Logistic Regression Functions"
  
def hypothesis(x,w,b):
   #accepts input vector x, input weight vector w and bias b'''\n",
    hx = np.dot(x,w)+b
    return sigmoid(hx)

def sigmoid(h):
    return 1.0/(1.0 + np.exp(-1.0*h))
   
def error(y,x,w,b):
    m = x.shape[0]
    err = 0.0
    for i in range(m):
        hx = hypothesis(x[i],w,b)
        err += y[i]*np.log2(hx)+(1-y[i])*np.log2(1-hx)
    return err
  
def get_grad(x,w,b,y):
    grad_b = 0.0
    grad_w = np.zeros(w.shape)
    m = x.shape[0]
    for i in range(m):
        hx = hypothesis(x[i],w,b)
        grad_w += (y[i] - hx)*x[i]
        grad_b +=  (y[i]-hx)
        grad_w /=m
        grad_b /=m
    return [grad_w,grad_b]
 
def gradient_descent(x,y,w,b,learning_rate=0.01):
    err = error(y,x,w,b)
    [grad_w,grad_b] = get_grad(x,w,b,y)
    w = w + learning_rate*grad_w
    b = b + learning_rate*grad_b
    return err,w,b
 
def predict(x,w,b):
    confidence = hypothesis(x,w,b)
    if confidence<0.5:
        return 0
    else:
         return 1
def get_acc(x_tst,y_tst,w,b):
    y_pred = []
    for i in range(y_tst.shape[0]):
        p = predict(x_tst[i],w,b)
        y_pred.append(p)
        y_pred = np.array(y_pred)
    return  float((y_pred==y_tst).sum())/y_tst.shape[0]

loss=[]
acc=[]
W = 2*np.random.random((dfx.shape[1],))
b = 5*np.random.random()

for i in range(1000):
    l,W,b = gradient_descent(dfx,dfy,W,b,learning_rate=0.1)
    loss.append(l)
    
plt.plot(loss)
plt.ylabel("Negative of Log Likelihood")
plt.xlabel("Time")
plt.show()


# In[ ]:



plt.plot(acc)
plt.show()
print(acc[-1])
#"## Plotting a Hyperplane or a decision boundary "
plt.figure(0)
plt.scatter(dist_01[:,0],dist_01[:,1],label='Class 0')
plt.scatter(dist_02[:,0],dist_02[:,1],color='r',marker='^',label='Class 1')
plt.xlim(-5,10)
plt.ylim(-5,10)
plt.xlabel('x1')
plt.ylabel('x2')
x = xtest
y = -(W[0]*x + b)/W[1]
plt.plot(x,y,color='k')
plt.legend()
plt.show()

