from sklearn.neural_network import MLPClassifier
import pandas as pd 

traindata = pd.read_excel('traindata.xlsx', header= None, usecols= 'A:V')
testdata = pd.read_excel('testdata.xlsx', header= None, usecols='A:V')
train_labels = pd.read_excel('traindata.xlsx', header = None, usecols= 'W')
test_labels = pd.read_excel('testdata.xlsx', header = None, usecols= 'W')
mlp = MLPClassifier(hidden_layer_sizes=(20,15,12,10), activation='relu', solver='adam', max_iter=500)
# print(test_labels.shape)
mlp.fit(traindata, train_labels)
print(mlp.score(testdata,test_labels))
print(mlp.coefs_)