import numpy as np
import csv

train_in = './data/1024.csv'
test_in = './data/test.csv'
D = 2 ** 20  # number of weights use for each model

def data(path, traindata = False):
    for t, line in enumerate(open(path,'r')):
        # initialize our generator
        if t == 0: # We filter the header
            # create a static x,
            # so we don't have to construct a new x for every instance
            # x = [0] * 27
            x = [0] * 26 # add one bias term
            continue
        # sparse x

        for m, feat in enumerate(line.rstrip().split(',')):
            if m == 0:
                ID = int(feat)
            elif traindata and m == 1:
                y = [float(feat)]
            else:
                # one-hot encode everything with hash trick
                # categorical: one-hotted
                # boolean: ONE-HOTTED
                # numerical: ONE-HOTTED!
                # note, the build in hash(), although fast is not stable,
                #       i.e., same value won't always have the same hash
                #       on different machines
                if traindata:
                    x[m - 1] = abs(hash(str(m - 1) + '_' + feat)) % D # training data from X[1]
                else:
                    x[m] = abs(hash(str(m) + '_' + feat)) % D # test data from X[1]

        yield (ID, x, y) if traindata else (ID, x)

train_out = './data/1024_hash.csv'
test_out = './data/test_hash.csv'

# csvfile_out = open(train_out, 'w')
# data_out = csv.writer(csvfile_out)
# for ID, x, y in data(train_in, traindata=True):
#     x = np.append(y, x)
#     x = np.append(str(ID), x)
#     data_out.writerow(x)
# csvfile_out.close()
# del data_out
# print "%s is ready." % (train_out)

csvfile_out = open(test_out, 'w')
data_out = csv.writer(csvfile_out)
for ID, x in data(test_in):
    x = np.append(str(ID), x)
    data_out.writerow(x)
csvfile_out.close()
del data_out
print "%s is ready." % (test_out)
