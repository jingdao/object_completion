import sys
import time
import caffe
import os
if os.system('nvcc --version') == 0:
	caffe.set_mode_gpu()
else:
	caffe.set_mode_cpu()
from importWeights import *
from importViews import *
import matplotlib.cm as cm

class termcolors:
	normal = '\033[0m'
	green = '\033[92m'
	yellow = '\033[93m'
	blue = '\033[94m'
	red = '\033[91m'

debugSingleSample=True

network = Model("data/finetuned_model.txt")
features = numpy.load('data/table_features.npy')
partial_views = GridData('data/table_partial.data','data/table_labels.data')
complete_views = GridData('data/table_complete.data','data/table_labels.data')
solver = caffe.SGDSolver('architecture/Net3DReg_solver_deconv.prototxt')
print(termcolors.red+'initialized solver'+termcolors.normal)
batchsize = 2;
solver.net.blobs['data'].reshape(batchsize,1200)
solver.test_nets[0].blobs['data'].reshape(batchsize,1200)
solver.net.blobs['label'].reshape(batchsize,1,30,30,30)
solver.test_nets[0].blobs['label'].reshape(batchsize,1,30,30,30)

print(termcolors.blue+'assign weights'+termcolors.normal)
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

trainIndices = []
testIndices = []
while (len(testIndices) < batchsize):
	r = numpy.random.randint(complete_views.num_samples)
	if not r in testIndices:
		testIndices.append(r)
if debugSingleSample:
	while (len(trainIndices) < batchsize):
		r = numpy.random.randint(complete_views.num_samples)
		if not r in trainIndices:
			trainIndices.append(r)
else:
	for i in range(complete_views.num_samples):
		if not i in testIndices:
			trainIndices.append(i)
numTraining = len(trainIndices)
numTesting = len(testIndices)
for i in range(batchsize):
	inputData = features[testIndices[i],:]
	referenceData = complete_views.samples[testIndices[i]]
	solver.test_nets[0].blobs['data'].data[i,:] = inputData
	solver.test_nets[0].blobs['label'].data[i,0,:,:,:] = referenceData

print(termcolors.yellow+'forward pass '+str(numTraining)+' training '+str(numTesting)+' testing samples'+termcolors.normal)
niter=100
test_interval=20
train_loss = numpy.zeros(niter)
test_loss = numpy.zeros(niter/test_interval)
for it in range(niter):
	for j in range(batchsize):
		id = trainIndices[(it * batchsize + j) % numTraining]
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
	#print it,train_loss[it],end-start
	#for s in ['b_fc5','b_conv3','b_conv2','b_conv1']:
		#d = solver.test_nets[0].blobs[s].data
		#print s,numpy.min(d),numpy.mean(d),numpy.max(d)

fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(train_loss)
plt.subplot(2,1,2)
plt.plot(test_loss)
plt.show()

def initialize_missing(data):
	mask = data < 0
#	filler = numpy.random.rand(len(data[mask])) > 0.9
	filler = numpy.ones(len(data[mask])) * 0.5
	res = numpy.array(data,dtype=numpy.float32)
	res[mask] = filler
	return res

for j in range(batchsize):
	fig = plt.figure()
	output = solver.test_nets[0].blobs['act_b_conv1'].data[j,0,:,:,:]
	f = features[testIndices[j],:]
	ax = fig.add_subplot(131, projection='3d')
	ax.set_xlim(0,30)
	ax.set_ylim(0,30)
	ax.set_zlim(0,30)
	src = initialize_missing(partial_views.samples[testIndices[j]])
	x,y,z = numpy.nonzero(src)
	color = src[x,y,z] * 2 - 1 
	ax.scatter(x,y,z,c=cm.jet(color),s=10)
	ax = fig.add_subplot(132, projection='3d')
	ax.set_xlim(0,30)
	ax.set_ylim(0,30)
	ax.set_zlim(0,30)
	x,y,z = numpy.nonzero(output>0.1)
	color = output[x,y,z]
	ax.scatter(x,y,z,c=cm.jet(color),s=10)
	ax = fig.add_subplot(133, projection='3d')
	ax.set_xlim(0,30)
	ax.set_ylim(0,30)
	ax.set_zlim(0,30)
	x,y,z = numpy.nonzero(complete_views.samples[testIndices[j]])
	ax.scatter(x,y,z,c='r',s=10)
	plt.title('Test sample '+str(j)+': index '+str(testIndices[j]))
	plt.show()
	
solver.net.blobs['data'].reshape(1,1200)
solver.net.blobs['label'].reshape(1,1,30,30,30)
for j in range(numTraining):
	fig = plt.figure()
	solver.net.blobs['data'].data[0,:] = features[trainIndices[j],:]
	solver.net.forward()
	output = solver.net.blobs['act_b_conv1'].data[0,0,:,:,:]
	ax = fig.add_subplot(131, projection='3d')
	ax.set_xlim(0,30)
	ax.set_ylim(0,30)
	ax.set_zlim(0,30)
	src = initialize_missing(partial_views.samples[trainIndices[j]])
	x,y,z = numpy.nonzero(src)
	color = src[x,y,z] * 2 - 1 
	ax.scatter(x,y,z,c=cm.jet(color),s=10)
	ax = fig.add_subplot(132, projection='3d')
	ax.set_xlim(0,30)
	ax.set_ylim(0,30)
	ax.set_zlim(0,30)
	x,y,z = numpy.nonzero(output>0.1)
	color = output[x,y,z]
	ax.scatter(x,y,z,c=cm.jet(color),s=10)
	ax = fig.add_subplot(133, projection='3d')
	ax.set_xlim(0,30)
	ax.set_ylim(0,30)
	ax.set_zlim(0,30)
	x,y,z = numpy.nonzero(complete_views.samples[trainIndices[j]])
	ax.scatter(x,y,z,c='r',s=10)
	plt.title('Train sample '+str(j)+': index '+str(trainIndices[j]))
	plt.show()
