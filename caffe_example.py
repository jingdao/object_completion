import sys
caffe_root = '/home/jd/Downloads/caffe/'  # this file is expected to be $
sys.path.insert(0, caffe_root + 'python')
import caffe
caffe.set_mode_cpu()
#from caffe import layers as L, params as P
from importWeights import *
from importViews import *

network = Model("discriminative_10_class.txt")
print network

partial_views = GridData('partial_view.data','labels.data')
complete_views = GridData('complete_view.data','labels.data')
print partial_views
print complete_views

class termcolors:
	normal = '\033[0m'
	green = '\033[92m'
	yellow = '\033[93m'
	blue = '\033[94m'
	red = '\033[91m'

solver = caffe.SGDSolver('3Dshapenet_solver.prototxt')
print(termcolors.red+'initialized solver'+termcolors.normal)
print(termcolors.green+'assign data'+termcolors.normal)
# assign data

batchsize = 20;
solver.net.blobs['data'].reshape(batchsize,1,30,30,30)
solver.net.blobs['label'].reshape(batchsize)

def initialize_missing(data):
	mask = data < 0
	filler = numpy.random.rand(len(data[mask]))
	res = data.copy()
	res[mask] = filler
	return res

label = -1
j = 0
for i in range(batchsize/2):
	while partial_views.labels[j] == label:
		j += 1
	label = partial_views.labels[j]
#	solver.net.blobs['data'].data[i,0,:,:,:] = initialize_missing(partial_views.samples[j])
	solver.net.blobs['data'].data[i,0,:,:,:] = complete_views.samples[j]
	solver.net.blobs['label'].data[2*i] = label
	j += 1
	solver.net.blobs['data'].data[i,0,:,:,:] = complete_views.samples[j]
	solver.net.blobs['label'].data[2*i+1] = label
	j += 1
print solver.net.blobs['label'].data

# initialize wights and bias
# Load parameters
print(termcolors.blue+'assign weights'+termcolors.normal)
print(termcolors.blue+'matlab weights'+termcolors.normal)
print network.layers[1].w.shape
print network.layers[2].w.shape
print network.layers[3].w.shape
print network.layers[4].w.shape
print network.layers[5].w.shape
print(termcolors.blue+'python weights'+termcolors.normal)
print solver.net.params['conv1'][0].data.shape
print solver.net.params['conv2'][0].data.shape
print solver.net.params['conv3'][0].data.shape
print solver.net.params['fc5'][0].data.shape
print solver.net.params['fc6'][0].data.shape

for i in range(network.layers[1].w.shape[0]):
	solver.net.params['conv1'][0].data[i,0,:,:,:] = network.layers[1].w[i,:,:,:].transpose()
solver.net.params['conv1'][1].data[...] = network.layers[1].c[:,0]
for i in range(network.layers[2].w.shape[0]):
	solver.net.params['conv2'][0].data[i,:,:,:,:] = network.layers[2].w[i,:,:,:,:].transpose()
solver.net.params['conv2'][1].data[...] = network.layers[2].c[:,0]
for i in range(network.layers[3].w.shape[0]):
	solver.net.params['conv3'][0].data[i,:,:,:,:] = network.layers[3].w[i,:,:,:,:].transpose()
solver.net.params['conv3'][1].data[...] = network.layers[3].c[:,0]
solver.net.params['fc5'][0].data[...] = network.layers[4].w.transpose()
solver.net.params['fc5'][1].data[...] = network.layers[4].c[0,:]
solver.net.params['fc6'][0].data[...] = network.layers[5].w.transpose()
solver.net.params['fc6'][1].data[...] = network.layers[5].c[0,:]

#for i in range(network.layers[1].w.shape[0]):
#	solver.net.params['conv1'][0].data[i,0,:,:,:] = network.layers[1].w[i,:,:,:]
#solver.net.params['conv1'][1].data[...] = network.layers[1].c[:,0]
#for i in range(network.layers[2].w.shape[0]):
#	for j in range(network.layers[2].w.shape[4]):
#		solver.net.params['conv2'][0].data[i,j,:,:,:] = network.layers[2].w[i,:,:,:,j]
#solver.net.params['conv2'][1].data[...] = network.layers[2].c[:,0]
#for i in range(network.layers[3].w.shape[0]):
#	for j in range(network.layers[3].w.shape[4]):
#		solver.net.params['conv3'][0].data[i,j,:,:,:] = network.layers[3].w[i,:,:,:,j]
#solver.net.params['conv3'][1].data[...] = network.layers[3].c[:,0]
#solver.net.params['fc5'][0].data[...] = network.layers[4].w.transpose()
#solver.net.params['fc5'][1].data[...] = network.layers[4].c[0,:]
#solver.net.params['fc6'][0].data[...] = network.layers[5].w.transpose()
#solver.net.params['fc6'][1].data[...] = network.layers[5].c[0,:]

print(termcolors.yellow+'forward pass'+termcolors.normal)
loss = solver.net.forward()
print loss
act_fc6 = solver.net.blobs['act_fc6'].data
prediction = numpy.argmax(act_fc6,axis=1)
print 'prediction ',prediction
