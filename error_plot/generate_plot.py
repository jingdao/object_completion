import numpy as np
import matplotlib.pyplot as plt

def sample(input, step):
    return [input[i] for i in range(len(input)) if i % step == 0]

test1200 = np.load('testLoss1200.npy')
train1200 = np.load('trainLoss1200.npy')
te1200 = sample(test1200, 2000)
tr1200 = sample(train1200, 2000)

test12000 = np.load('testLoss12000.npy')
train12000 = np.load('trainLoss12000.npy')
te12000 = sample(test12000, 2000)
tr12000 = sample(train12000, 2000)

test1200_1200 = np.load('testLoss1200_1200.npy')
train1200_1200 = np.load('trainLoss1200_1200.npy')
te1200_1200 = [test1200_1200[i] for i in range(len(test1200_1200)) if i % 100 == 0]
tr1200_1200 = sample(train1200_1200, 2000)

epochs = [i for i in range(len(test1200)) if i % 2000 == 0]

#plot training errors
plt1200, = plt.plot(epochs, tr1200, '-', label='FC of size 1200')
plt12000, = plt.plot(epochs, tr12000, '-', label='FC of size 12000')
plt1200_1200, = plt.plot(epochs, tr1200_1200, '-', label='FC of size 1200*1200')
plt.legend(handles=[plt1200, plt12000, plt1200_1200])
plt.savefig('train.png')
plt.clf()

#plot test errors
plt1200, = plt.plot(epochs, te1200, '-', label='FC of size 1200')
plt12000, = plt.plot(epochs, te12000, '-', label='FC of size 12000')
plt1200_1200, = plt.plot(epochs, te1200_1200, '-', label='FC of size 1200*1200')
plt.legend(handles=[plt1200, plt12000, plt1200_1200])
plt.savefig('test.png')
plt.clf()
