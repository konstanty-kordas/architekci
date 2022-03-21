import numpy as np
import pandas as pd
import matplotlib as plt

iris = pd.read_csv('./IRIS.csv')
training = iris


def MSE(actual, expected):
    mse = np.mean(np.square(actual-expected))
    return mse

def sigmoid(x):
    return(1/(1 + np.exp(-x)))

weights = np.random.rand(5)


print(weights)

def single(set,W):
    results = []
    for elem in set:
        r = 0
        for i in range(4):
            r+=elem[i]*W[i]
        results.append(r)
    return results
    
# print(MSE(correctClassification,single(training,weights)))
