import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
#import 
def data_preprocessor(File_data):
  arr = []
  for i in range(1,2*int(File_data[0]),2):
    arr.append(np.array([File_data[i],File_data[i+1]]))

  arr = np.array(arr)
  arr = arr -np.array([min(arr[:,0]),min(arr[:,1])])

  # Now normalizing to one

  arr = arr /np.array([[max(arr[:,0]),max(arr[:,1])]])
  return arr


#print(len(os.listdir(r"dA/train")))
X_train = []
Y_train = []
for ele in os.listdir(r"Data/"):
    for folder in os.listdir(f"Data/{ele}/train"):
        data = np.loadtxt(f"Data/{ele}/train/{folder}",dtype = float)
        arr = data_preprocessor(data)
        X_train.append(arr)
        Y_train.append(ele)
#X_train , Y_train = list(X_train) , list(Y_train)

#idx = np.random.permutation(len(Y_train))

#X_train,Y_train = X_train[idx], Y_train[idx]
#print(Y_train)

with open('X_train.pickle', 'wb') as fh:
   pickle.dump(X_train, fh)

with open('Y_train.pickle', 'wb') as fh:
   pickle.dump(Y_train, fh)
