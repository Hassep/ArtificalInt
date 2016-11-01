"""
================================
Nearest Neighbors Classification
================================
Modified in class by Dr. Rivas
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from numpy import genfromtxt
from sklearn.model_selection import KFold
import time

#generates a random dataset 
def genDataSet(N):
    X = np.random.normal(0, 1, N)
    ytrue = (np.cos(X)+2) / (np.cos(X*1.4)+2)
    noise = np.random.normal(0, 0.2, N)
    y = ytrue + noise
    return X, y, ytrue


#impements the dataset 
X, y, ytrue = genDataSet(1000)
plt.plot(X,y,'.')
plt.plot(X,ytrue,'rx')
plt.show()
X = np.array(X).reshape(len(X),1)
# y[y<>0] = -1    #rest of numbers are negative class
# y[y==0] = +1    #number zero is the positive class

highest=0
bestk=[]
kc=0
for n_neighbors in range(1,900,2):
  kf = KFold(n_splits=10)
  #n_neighbors = 85
  kscore=[]
  k=0
  for train, test in kf.split(X):
    #print("%s %s" % (train, test))
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
  
    #time.sleep(100)
  
    # we create an instance of Neighbors Classifier and fit the data.
    clf = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
    clf.fit(X_train, y_train)
  
    kscore.append(abs(clf.score(X_test,y_test)))
    #print kscore[k]
    k=k+1
  
  print (n_neighbors)
  bestk.append(sum(kscore)/len(kscore))
  print bestk[kc]
  kc+=1


# to do here: given this array of E_outs in CV, find the max, its 
# corresponding index, and its corresponding value of n_neighbors
sbestk = sorted(bestk, reverse=True)
get_index = [sbestk[0],sbestk[1],sbestk[2]]

for i in get_index:
  print bestk.index(i),(bestk.index(i) + bestk.index(i) + 1)
print sbestk[0],sbestk[1],sbestk[2]
#print bestk