import pandas as pd
import numpy as np
NUM_HIDDEN_LAYER = 1
testdata = pd.read_excel('testdata.xlsx', header= None, usecols='A:V')
test_labels = pd.read_excel('testdata.xlsx', header = None, usecols= 'W')
# print(mlp.coefs_)
# print(type(mlp.coef))

W = []
b = []
for i in range(NUM_HIDDEN_LAYER):
    W.append(np.loadtxt('mlpcoefs{0}{1}.csv'.format(i + 1,i + 2),delimiter=','))
    k = np.loadtxt('mlpbias{0}{1}.csv'.format(i + 1, i + 2), delimiter=',')
    # print(k.shape)
    k = k.reshape((k.shape[0],1))
    b.append(k)
    # b[-1] = k
A = [testdata.to_numpy()]
for i in range(2):
    A.append(A[i].dot(W[i]) + b[i].reshape((b[i].shape[0],)))
    A[-1] = np.maximum(0, A[-1])
test_labels = test_labels.to_numpy()
print(np.mean(np.expand_dims(np.argmax(A[-1],axis=1),1) == test_labels))
