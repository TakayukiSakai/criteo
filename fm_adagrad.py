#!/usr/bin/pypy

from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from pymmh3 import hash
import numpy as np

# parameters #################################################################

train = 'train.txt'  # path to training file
test = 'test.txt'  # path to testing file

factor = 4

logbatch = 100000
dotest = True

D = 2 ** 24    # number of weights use for learning

signed = False    # Use signed hash? Set to False for to reduce number of hash calls

alpha = .05       # learning rate for sgd optimization

adapt = 1.        # Use adagrad, sets it as power of adaptive factor. >1 will amplify adaptive measure and vice versa
fudge = .5        # Fudge factor


header = ['Label','i1','i2','i3','i4','i5','i6','i7','i8','i9','i10','i11','i12','i13','c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','c11','c12','c13','c14','c15','c16','c17','c18','c19','c20','c21','c22','c23','c24','c25','c26']

# function definitions #######################################################

# A. Bounded logloss
# INPUT:
#     p: our prediction
#     y: real answer
# OUTPUT
#     logarithmic loss of p given y
def logloss(p, y):
    p = max(min(p, 1. - 10e-17), 10e-17)        # The bounds
    return -log(p) if y == 1. else -log(1. - p)


# B. Apply hash trick of the original csv row
# for simplicity, we treat both integer and categorical features as categorical
# INPUT:
#     csv_row: a csv dictionary, ex: {'Lable': '1', 'I1': '357', 'I2': '', ...}
#     D: the max index that we can hash to
# OUTPUT:
#     x: a list of indices that its value is 1
def get_x(csv_row, D):
    fullind = []
    for key, value in csv_row.items():
        s = key + '=' + value
        fullind.append(hash(s) % D) # weakest hash ever ?? Not anymore :P

    x = {}
    for index in fullind:
        if(not x.has_key(index)):
            x[index] = 0
        if signed:
            x[index] += (1 if (hash(str(index))%2)==1 else -1) # Disable for speed
        else:
            x[index] += 1
    
    return x  # x contains indices of features that have a value as number of occurences


# C. Get probability estimation on x
# INPUT:
#     x : features
#     w0: global bias
#     w : weights
#     v : weight matrix
# OUTPUT:
#     probability of p(y = 1 | x; w)
def get_p(x, w0, w, v):
    wTx = w0
    for i, xi in x.items():
        wTx += w[i] * xi  # w[i] * x[i]
    
    for f in range(factor):
        sumVjfXj = 0.
        sumV2X2  = 0.
        for i, xi in x.items():
            vx = v[i][f] * xi
            sumVjfXj += vx
            sumV2X2  += (vx * vx)
        
        wTx += 0.5 * (sumVjfXj * sumVjfXj - sumV2X2)
        
    return 1. / (1. + exp(-max(min(wTx, 50.), -50.)))  # bounded sigmoid


# D. Update given model
# INPUT:
#     w0: global bias
#     w : weights
#     v : weight matrix
#     n : a counter that counts the number of times we encounter a feature
#        this is used for adaptive learning rate
#     x : feature
#     p : prediction of our model
#     y : answer
# OUTPUT:
#     w: updated model
#     n: updated count
def update_w(w0, w, v, g0, g, x, p, y):
    # alpha / (sqrt(g) + 1) is the adaptive learning rate heuristic
    # (p - y) * x[i] is the current gradient
    # note that in our case, if i in x then x[i] = 1
    delta = p - y
    if adapt > 0:
        g0 += delta ** 2
    w0 -= delta * alpha / (sqrt(g0) ** adapt)
    
    sumVfx = [0.] * factor
    for i, xi in x.items():
        for f in range(factor):
            sumVfx[f] += v[i][f] * xi
    
    for i, xi in x.items():
        delta = (p - y) * xi
        if adapt > 0:
            g[i] += delta ** 2
            
        w[i] -= delta * alpha / (sqrt(g[i]) ** adapt)  # Minimising log loss
        
        for f in range(factor):
            h = xi * (sumVfx[f] - v[i][f] * xi)
            v[i][f] -= (p - y) * h * alpha / (sqrt(g[i]) ** adapt)
            
    return w0, w, v, g0, g


# training and testing #######################################################

# initialize our model
w0 = -1.
w = [0.] * D  # weights
v = [np.random.normal(0, 0.05, factor) for y in range(D)] # symmetry breaking
g0 = fudge
g = [fudge] * D  # sum of historical gradients

# start training a logistic regression model using on pass sgd
loss = 0.
lossb = 0.
for t, row in enumerate(DictReader(open(train), header, delimiter='\t')):
    y = 1. if row['Label'] == '1' else 0.

    del row['Label']  # can't let the model peek the answer

    # main training procedure
    # step 1, get the hashed features
    x = get_x(row, D)
    # step 2, get prediction
    p = get_p(x, w0, w, v)

    # for progress validation, useless for learning our model
    lossx = logloss(p, y)
    loss += lossx
    lossb += lossx
    if t % logbatch == 0 and t > 1:
        print('%s\tencountered: %d\tcurrent whole logloss: %f\tcurrent batch logloss: %f' % (datetime.now(), t, loss/t, lossb/logbatch))
        lossb = 0.

    # step 3, update model with answer
    w0, w, v, g0, g = update_w(w0, w, v, g0, g, x, p, y)

if not dotest:
    exit()

# testing (build kaggle's submission file)
with open('submission.csv', 'w') as submission:
    submission.write('Id,Predicted\n')
    for t, row in enumerate(DictReader(open(test), header[1:], delimiter='\t')):
        x = get_x(row, D)
        p = get_p(x, w0, w, v)
        submission.write('%d,%f\n' % (60000000+int(t), p))
