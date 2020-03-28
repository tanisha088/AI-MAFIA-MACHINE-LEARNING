import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def drawing(sample):
    img = sample.reshape((28, 28))
    plt.imshow(img, cmap='gray')
    plt.show()


def distance(x,y):
    return np.sqrt((sum((x - y)**2)))


def knn(x1, y1, query_point, k=5):
    vals = []
    m = x1.shape[0]
    for i in range(m):
        d = distance(query_point, x1[i])
        vals.append((d,y1[i]))
    vals=sorted(vals)
    vals=vals[:k]
    vals=np.array(vals)
    new_vals = np.unique(vals[:,1],return_counts=True)
    index = new_vals[1].argmax()
    pred =  new_vals[0][index]
    return pred


df = pd.read_csv('./train_knn.csv')
data = df.values
x = data[:, 1:]
y = data[:, 0]
print(x)
print(y)
split = int(0.8*x.shape[0])
x_train = x[:split, :]
y_train = y[:split]
x_test = x[split:, :]
print("the work")
n = knn(x_train, y_train, x_test[0])
print("the results are")
print(n)
drawing(x_test[1])




