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

def initialize_missing(data):
	mask = data < 0
	filler = numpy.ones(len(data[mask])) * 0.5
	res = numpy.array(data,dtype=numpy.float32)
	res[mask] = filler
	return res

network = Model("data/finetuned_model.txt")
partial_views = GridData('data/table_partial.data','data/table_labels.data')
complete_views = GridData('data/table_complete.data','data/table_labels.data')
solver_path = sys.argv[1]
solver = caffe.AdamSolver(solver_path)
solver.net.blobs['data'].reshape(1,1,30,30,30)
solver.net.blobs['label'].reshape(1,1,30,30,30)
path_net = sys.argv[2]
solver.net.copy_from(path_net)

for j in range(10):
	id = numpy.random.randint(complete_views.num_samples)
	fig = plt.figure()
	src = initialize_missing(partial_views.samples[id])
	solver.net.blobs['data'].data[0,0,:,:,:] = src
	solver.net.forward()
	output = solver.net.blobs['act_b_conv1'].data[0,0,:,:,:]
	ax = fig.add_subplot(131, projection='3d')
	ax.set_xlim(0,30)
	ax.set_ylim(0,30)
	ax.set_zlim(0,30)
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
	x,y,z = numpy.nonzero(complete_views.samples[id])
	ax.scatter(x,y,z,c='r',s=10)
	plt.title('Completion sample '+str(j)+': index '+str(id))
        fig.savefig('completion.png')
