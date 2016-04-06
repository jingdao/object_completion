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
batchsize = 1;
solver.net.blobs['data'].reshape(batchsize,1,1200)
solver.net.blobs['label'].reshape(batchsize,1,30,30,30)

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

print(termcolors.yellow+'forward pass'+termcolors.normal)
solver.net.blobs['data'].data[0,0,:] = features[0,:]
solver.net.blobs['label'].data[0,0,:,:,:] = complete_views.samples[0]
loss = solver.net.forward()
output = solver.net.blobs['act_b_conv1'].data[0,0,:,:,:]
print loss
plotObject(output)
