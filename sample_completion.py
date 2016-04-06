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

def comparePlots(obj1,obj2,obj3):
	fig = plt.figure()
	ax = fig.add_subplot(131, projection='3d')
	x,y,z = numpy.nonzero(obj1>0)
	ax.scatter(x,y,z,c='r',s=10)
	ax = fig.add_subplot(132, projection='3d')
	x,y,z = numpy.nonzero(obj2>0)
	ax.scatter(x,y,z,c='r',s=10)
	ax = fig.add_subplot(133, projection='3d')
	x,y,z = numpy.nonzero(obj3>0)
	ax.scatter(x,y,z,c='r',s=10)
	plt.show()

network = Model("finetuned_model.txt")
partial_views = GridData('partial_view.data','labels.data')
complete_views = GridData('complete_view.data','labels.data')
solver = caffe.SGDSolver('Net3DReg_solver.prototxt')
print(termcolors.red+'initialized solver'+termcolors.normal)
batchsize = 1;
solver.net.blobs['data'].reshape(batchsize,1,30,30,30)
solver.net.blobs['label'].reshape(batchsize,1,30,30,30)

print(termcolors.blue+'assign weights'+termcolors.normal)
for i in range(network.layers[1].w.shape[0]):
	solver.net.params['conv1'][0].data[i,0,:,:,:] = network.layers[1].w[i,:,:,:]
solver.net.params['conv1'][1].data[...] = network.layers[1].c[:,0]
for i in range(network.layers[2].w.shape[0]):
	for j in range(network.layers[2].w.shape[4]):
		solver.net.params['conv2'][0].data[i,j,:,:,:] = network.layers[2].w[i,:,:,:,j]
solver.net.params['conv2'][1].data[...] = network.layers[2].c[:,0]
for i in range(network.layers[3].w.shape[0]):
	for j in range(network.layers[3].w.shape[4]):
		solver.net.params['conv3'][0].data[i,j,:,:,:] = network.layers[3].w[i,:,:,:,j]
solver.net.params['conv3'][1].data[...] = network.layers[3].c[:,0]
solver.net.params['fc5'][0].data[...] = network.layers[4].w.transpose()
solver.net.params['fc5'][1].data[...] = network.layers[4].c[0,:]
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

def initialize_missing(data):
	mask = data < 0
	filler = numpy.random.rand(len(data[mask])) > 0.9
	res = data.copy()
	res[mask] = filler
	return res

print(termcolors.yellow+'forward pass'+termcolors.normal)
inputData = initialize_missing(partial_views.samples[0])
referenceData = complete_views.samples[0]
solver.net.blobs['data'].data[0,0,:,:,:] = inputData
solver.net.blobs['label'].data[0,0,:,:,:] = referenceData
loss = solver.net.forward()
outputData = solver.net.blobs['act_b_conv1'].data[0,0,:,:,:]
print loss
comparePlots(inputData,outputData,referenceData)
