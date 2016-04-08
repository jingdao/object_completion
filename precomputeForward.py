import sys
import time
import caffe
caffe.set_mode_cpu()
from importWeights import *
from importViews import *

network = Model("finetuned_model.txt")
partial_views = GridData('partial_view.data','labels.data')

solver = caffe.SGDSolver('3Dshapenet_solver.prototxt')
batchsize = 10;
solver.net.blobs['data'].reshape(batchsize,1,30,30,30)
solver.net.blobs['label'].reshape(batchsize)
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
#solver.net.params['fc6'][0].data[...] = network.layers[5].w.transpose()
#solver.net.params['fc6'][1].data[...] = network.layers[5].c[0,:]

def initialize_missing(data):
	mask = data < 0
	filler = numpy.random.rand(len(data[mask])) > 0.1
	res = data.copy()
	res[mask] = filler
	return res

start = time.clock()
j=0
features=None
while j < partial_views.num_samples:
	for i in range(batchsize):
		solver.net.blobs['data'].data[i,0,:,:,:] = initialize_missing(partial_views.samples[j].transpose())
		j += 1
		if j >= partial_views.num_samples:
			break
	solver.net.forward()
	if not features is None:
		features = numpy.vstack((features,solver.net.blobs['act_fc5'].data[:i+1,:]))
	else:
		features = numpy.copy(solver.net.blobs['act_fc5'].data)
end = time.clock()
numpy.save('features.npy',features)
print('forward pass for '+str(partial_views.num_samples)+' samples (batch='+str(batchsize)+') took '+str(end-start)+' seconds')
