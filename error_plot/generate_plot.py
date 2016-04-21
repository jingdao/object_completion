import numpy as np
import matplotlib.pyplot as plt

def sample(input, step):
    return [input[i] for i in range(len(input)) if i % step == 0]

step = 2000

test1200 = np.load('testLoss1200.npy')
train1200 = np.load('trainLoss1200.npy')
te1200 = sample(test1200, step)
tr1200 = sample(train1200, step)

test12000 = np.load('testLoss12000.npy')
train12000 = np.load('trainLoss12000.npy')
te12000 = sample(test12000, step)
tr12000 = sample(train12000, step)

test1200_1200 = np.load('testLoss1200_1200.npy')
train1200_1200 = np.load('trainLoss1200_1200.npy')
te1200_1200 = [test1200_1200[i] for i in range(len(test1200_1200)) if i % 100 == 0]
tr1200_1200 = sample(train1200_1200, step)

test1200mask = np.load('testLoss1200mask.npy')
train1200mask = np.load('trainLoss1200mask.npy')
te1200mask = [test1200mask[i] for i in range(len(test1200mask)) if i % 100 == 0]
tr1200mask = sample(train1200mask, step)

epochs = [i for i in range(len(test1200)) if i % step == 0]

#plot training errors
plt1200, = plt.plot(epochs, np.log(tr1200), '-', label='FC size of 1200')
plt12000, = plt.plot(epochs, np.log(tr12000), '-', label='FC size of 12000')
plt1200_1200, = plt.plot(epochs, np.log(tr1200_1200), '-', label='FC size of 1200*1200')
plt1200mask, = plt.plot(epochs, np.log(tr1200mask), '-', label='FC size of 1200 with mask')
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error(ln)')
plt.legend(handles=[plt1200, plt12000, plt1200_1200, plt1200mask])
plt.savefig('train.png')
plt.clf()

#plot test errors
plt1200, = plt.plot(epochs, np.log(te1200), '-', label='FC size of 1200')
plt12000, = plt.plot(epochs, np.log(te12000), '-', label='FC size of 12000')
plt1200_1200, = plt.plot(epochs, np.log(te1200_1200), '-', label='FC size of 1200*1200')
plt1200mask, = plt.plot(epochs, np.log(te1200mask), '-', label='FC size of 1200 with mask')
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error(ln)')
plt.legend(handles=[plt1200, plt12000, plt1200_1200, plt1200mask])
plt.savefig('test.png')
plt.clf()
