from __future__ import division
from datetime import datetime
from math import log, exp, sqrt
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier

train_path = './data/train_out3.csv'
test_path = './data/test_out3.csv'

D = 2 ** 20  # number of weights use for each model

def data(path, traindata=False):
    for t, line in enumerate(open(path, 'r')):
        # initialize our generator
        if t == 0: # We filter the header
            x = [0] * 22  # add one bias term
            continue

        for m, feat in enumerate(line.rstrip().split(',')):
            if m == 0:
                ID = int(feat)
            elif traindata and m == 1:
                y = [float(feat)]
            else:
                if traindata:
                    x[m - 1] = abs(hash(str(m - 1) + '_' + feat)) % D # training data from x[1]
                else:
                    x[m] = abs(hash(str(m) + '_' + feat)) % D # test data from x[1]

        yield (ID, x, y) if traindata else (ID, x)

def importdata(path):
    train_x = []
    train_y = []
    tt = 0
    holdout = 40 # Around one million samples
    for ID, x, y in data(path, traindata=True):
        randomstart = random.sample(range(holdout), 1)[0]
        if (holdout and (tt - randomstart) % holdout == 0):
            train_x += x
            train_y += y
            tt += 1
    train_x = np.array(train_x).reshape(tt, -1)
    train_y = np.array(train_y).reshape(tt, -1)
    return train_x, train_y

def subsampling(train_x, train_y, numberSamples=100000):
    tt = 0
    rownumber = random.sample(range(len(train_y)), numberSamples)
    for i in range(numberSamples):
        yield (train_x[rownumber[tt]], train_y[rownumber[tt]])
        tt += 1

# main learning process ######################################################
start = datetime.now()

numberSamples = 100000
numberTrees = 10
depthTrees = 5
jobsTrees = 2
iteration = 10
clf_tree = np.array([])

train_x, train_y = importdata(train_path)
print "Importing training data is finished."
for i in range(iteration):
    x_tree = []
    y_tree = []
    for x, y in subsampling(train_x, train_y, numberSamples):
        x_tree = np.append(x_tree, x)
        y_tree = np.append(y_tree, y)
    x_tree = np.array(x_tree).reshape(numberSamples, -1)
    y_tree = np.array(y_tree).reshape(numberSamples, -1)

    clf = RandomForestClassifier(n_estimators=numberTrees, max_depth=depthTrees ,n_jobs=jobsTrees)
    clf.fit(x_tree, np.ravel(y_tree))
    clf_tree = np.append(clf_tree, clf)
    print "Random Forest %s is ready." % (str(i))

del train_x, train_y, x_tree, y_tree

print "Start writing submission file."
with open('submission2-.csv', 'w') as outfile:
    outfile.write('id,click\n')
    for ID, x in data(test_path):
        p = np.empty(iteration * numberTrees)
        for i in range(iteration * numberTrees):
            p[i] = clf_tree[i].predict_proba(np.array([x]))[0][1]
        outfile.write('%s,%s\n' % (ID, str(np.mean(p))))





