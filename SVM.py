from __future__ import division
import numpy as np
from numpy import genfromtxt
from sklearn import svm
from sklearn import preprocessing

data = genfromtxt('./data/sample_0.csv', delimiter=',')
data = np.array(data)
X = data[:, 2:]
Y = data[:, 0]
Y -= Y == 0
clf = svm.SVC(gamma=6 ** 1.5, cache_size=4000, class_weight='auto')
clf.fit(X, Y)
Y_hat = clf.predict(X)

base_line = sum(Y == 1) / Y.shape[0]
print base_line
error_rate = np.sum(Y_hat != Y) / Y.shape[0]
print(error_rate)

