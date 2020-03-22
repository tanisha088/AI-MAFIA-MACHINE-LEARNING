#!/usr/bin/env python
# coding: utf-8

# In[44]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dfx = pd.read_csv('C:/Users/user/Desktop/PYCHARM_PROJECTS/Linear_X_Train1.csv')
dfy = pd.read_csv('C:/Users/user/Desktop/PYCHARM_PROJECTS/Linear_Y_Train1.csv')
dftest = pd.read_csv('C:/Users/user/Desktop/PYCHARM_PROJECTS/Linear_X_Test1.csv')
x = dfx.values
y = dfy.values
xtest = dftest.values

x = x.reshape((-1, 1))
y = y.reshape((-1, 1))
X = (x - x.mean()) / x.std()
Y = y

plt.scatter(X, Y)
plt.show()

from sklearn.linear_model import LinearRegression

model = LinearRegression()

## Training

model.fit(X, Y)

# Predictions

output = model.predict(xtest)
#print(output)

# Parameters Learned
bias = model.intercept_
coeff = model.coef_

print(bias)
print(coeff)

## Score

model.score(X, Y)
model.score(xtest,output)

## Visualise
plt.scatter(X, Y, label='data')
m = plt.plot(xtest, output, color='orange', label='prediction')
plt.legend()
plt.show()


print(output)


# In[ ]:




