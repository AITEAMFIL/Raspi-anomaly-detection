import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import time
testset = pd.read_excel('testdata.xlsx', header=None)
X1 = pd.read_excel('traindata.xlsx', header=None).iloc[:,:-1].astype(np.float).to_numpy()
for h in range(6, 47):
    time1 = time.time()
    X = testset.iloc[:, :-1].astype(np.float).to_numpy().T
    y = testset.iloc[:, -1].astype(np.int).to_numpy().T
    pca = PCA(n_components=6, svd_solver='full')
    W1 = np.loadtxt('mlp_W1{}.csv'.format(h), delimiter=',')
    W2 = np.loadtxt('mlp_W2{}.csv'.format(h), delimiter=',')
    b1 = np.loadtxt('mlp_b1{}.csv'.format(h), delimiter=',')
    b2 = np.loadtxt('mlp_b2{}.csv'.format(h), delimiter=',')
    pca.fit(X1)
    XPCA = pca.transform(X.T).T

    b1 = b1.reshape((b1.shape[0], 1))
    b2 = b2.reshape((b2.shape[0], 1))
    Z1 = np.dot(W1.T, XPCA) + b1
    A1 = np.maximum(Z1, 0)
    Z2 = np.dot(W2.T, A1) + b2
    predicted_class = np.argmax(Z2, axis=0)
    testing_accuracy = np.mean(predicted_class == y)
    print('testing accuracy: %.2f %%' % (100*testing_accuracy))
    test_time = time.time() - time1
    print('test time: ', test_time)
    with open('pi_test_mlp.txt', 'a') as f:
        f.write(str(h) + "Test time: " + str(test_time) + '\n')
        f.write(str(h) + "Accuracy: " + str(testing_accuracy) + '\n')
        f.close()
    time.sleep(15)

