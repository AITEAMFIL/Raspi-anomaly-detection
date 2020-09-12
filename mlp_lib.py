from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np

traindata = pd.read_excel('trainset.xlsx', header= None, usecols= 'A:V')
testdata = pd.read_excel('testset.xlsx', header= None, usecols='A:V')
train_labels = pd.read_excel('trainset.xlsx', header = None, usecols= 'W')
test_labels = pd.read_excel('testset.xlsx', header = None, usecols= 'W')
mlp = MLPClassifier(hidden_layer_sizes=(40), activation='relu', solver='adam', max_iter=500)
# print(test_labels.shape)
mlp.fit(traindata, train_labels)
print(mlp.score(testdata,test_labels))
# print(mlp.coefs_)
# print(type(mlp.coef))
for i in range(len(mlp.coefs_)):
    np.savetxt('mlpcoefs{0}{1}.csv'.format(i + 1,i + 2),mlp.coefs_[i],delimiter=",")
    np.savetxt('mlpbias{0}{1}.csv'.format(i + 1, i + 2), mlp.intercepts_[i],delimiter=",")
