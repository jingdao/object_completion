import sys
import time
import caffe
caffe.set_mode_cpu()
from importWeights import *
from importViews import *

class termcolors:
	normal = '\033[0m'
	green = '\033[92m'
	yellow = '\033[93m'
	blue = '\033[94m'
	red = '\033[91m'

network = Model("finetuned_model.txt")
features = numpy.load('features.npy')
complete_views = GridData('complete_view.data','labels.data')
solver = caffe.SGDSolver('Net3DReg_solver_deconv.prototxt')
print(termcolors.red+'initialized solver'+termcolors.normal)
batchsize = 5;
solver.net.blobs['data'].reshape(batchsize,1200)
solver.test_nets[0].blobs['data'].reshape(batchsize,1200)
solver.net.blobs['label'].reshape(batchsize,1,30,30,30)
solver.test_nets[0].blobs['label'].reshape(batchsize,1,30,30,30)

print(termcolors.blue+'assign weights'+termcolors.normal)
print(termcolors.blue+'matlab weights'+termcolors.normal)
print network.layers[4].dw.shape
print network.layers[4].b.shape
print network.layers[3].dw.shape
print network.layers[3].b.shape
print network.layers[2].dw.shape
print network.layers[2].b.shape
print network.layers[1].dw.shape
print network.layers[1].b.shape
print(termcolors.blue+'python weights'+termcolors.normal)
print solver.net.params['b_fc5'][0].data.shape
print solver.net.params['b_fc5'][1].data.shape
print solver.net.params['b_conv3'][0].data.shape
print solver.net.params['b_conv3'][1].data.shape
print solver.net.params['b_conv2'][0].data.shape
print solver.net.params['b_conv2'][1].data.shape
print solver.net.params['b_conv1'][0].data.shape
print solver.net.params['b_conv1'][1].data.shape

solver.net.params['b_fc5'][0].data[...] = network.layers[4].dw
solver.net.params['b_fc5'][1].data[...] = network.layers[4].b[0,:]
for i in range(network.layers[3].dw.shape[0]):
	for j in range(network.layers[3].dw.shape[4]):
		solver.net.params['b_conv3'][0].data[i,j,:,:,:] = network.layers[3].dw[i,:,:,:,j]
b = network.layers[3].b
solver.net.params['b_conv3'][1].data[...] = b.transpose().reshape((b.shape[3],-1)).mean(axis=1)
for i in range(network.layers[2].dw.shape[0]):
	for j in range(network.layers[2].dw.shape[4]):
		solver.net.params['b_conv2'][0].data[i,j,:,:,:] = network.layers[2].dw[i,:,:,:,j]
b = network.layers[2].b
solver.net.params['b_conv2'][1].data[...] = b.transpose().reshape((b.shape[3],-1)).mean(axis=1)
for i in range(network.layers[1].dw.shape[0]):
	solver.net.params['b_conv1'][0].data[i,0,:,:,:] = network.layers[1].dw[i,:,:,:]
b = network.layers[1].b
solver.net.params['b_conv1'][1].data[...] = b.transpose().reshape((1,-1)).mean()


numTraining = complete_views.num_samples - batchsize
for i in range(batchsize):
	inputData = features[numTraining+i,:]
	referenceData = complete_views.samples[numTraining+i]
	solver.test_nets[0].blobs['data'].data[i,:] = inputData
	solver.test_nets[0].blobs['label'].data[i,0,:,:,:] = referenceData

print(termcolors.yellow+'forward pass '+str(numTraining)+' training '+str(complete_views.num_samples-numTraining)+' testing samples'+termcolors.normal)
niter=100
test_interval=20
train_loss = numpy.zeros(niter)
test_loss = numpy.zeros(niter/test_interval)
for it in range(niter):
	for j in range(batchsize):
		id = (it * batchsize + j) % numTraining
		inputData = features[id,:]
		referenceData = complete_views.samples[id]
		solver.net.blobs['data'].data[j,:] = inputData
		solver.net.blobs['label'].data[j,0,:,:,:] = referenceData
	start = time.clock()
	solver.step(1)  # SGD by Caffe
	end = time.clock()
    
    # store the train loss
	train_loss[it] = solver.net.blobs['loss'].data
	if it % test_interval == 0:
		test_loss[it/test_interval] = solver.test_nets[0].blobs['loss'].data
	print it,train_loss[it],end-start

plt.subplot(2,1,1)
plt.plot(train_loss)
plt.subplot(2,1,2)
plt.plot(test_loss)
plt.show()
output = solver.net.blobs['act_b_conv1'].data[0,0,:,:,:]
fig = plt.figure()
ax = fig.add_subplot(131, projection='3d')
x,y,z = numpy.nonzero(output > 0.5)
ax.scatter(x,y,z,c='r',s=10)
plt.show()
