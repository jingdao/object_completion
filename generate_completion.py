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

classname = 'monitor'
partial_views = GridData('data/test_separate_class_correct_view/'+classname+'_partial.data','data/test_separate_class_correct_view/'+classname+'_labels.data')
complete_views = GridData('data/test_separate_class_correct_view/'+classname+'_complete.data','data/test_separate_class_correct_view/'+classname+'_labels.data')
solver_path = sys.argv[1]
solver = caffe.AdamSolver(solver_path)
solver_single_class = caffe.AdamSolver(solver_path)
use_mask = 'mask' in solver.net.blobs.keys()
solver.net.blobs['data'].reshape(1,1,30,30,30)
solver.net.blobs['label'].reshape(1,1,30,30,30)
solver_single_class.net.blobs['data'].reshape(1,1,30,30,30)
solver_single_class.net.blobs['label'].reshape(1,1,30,30,30)
if use_mask:
	solver.net.blobs['visible'].reshape(1,1,30,30,30)
	solver.net.blobs['mask'].reshape(1,1,30,30,30)
	solver_single_class.net.blobs['visible'].reshape(1,1,30,30,30)
	solver_single_class.net.blobs['mask'].reshape(1,1,30,30,30)
path_net = sys.argv[2]
solver.net.copy_from(path_net)
solver_single_class.net.copy_from(sys.argv[3])

for j in range(1):
	#id = numpy.random.randint(complete_views.num_samples)
	id = 249
	src = initialize_missing(partial_views.samples[id])
	solver.net.blobs['data'].data[0,0,:,:,:] = src
	solver_single_class.net.blobs['data'].data[0,0,:,:,:] = src
	visible = numpy.array(src == 1,dtype=numpy.float)
	mask = numpy.array(src == 0.5,dtype=numpy.float)
	if use_mask:
		solver.net.blobs['visible'].data[0,0,:,:,:] = visible
		solver.net.blobs['mask'].data[0,0,:,:,:] = mask
		solver_single_class.net.blobs['visible'].data[0,0,:,:,:] = visible
		solver_single_class.net.blobs['mask'].data[0,0,:,:,:] = mask
	solver.net.forward()
	solver_single_class.net.forward()
	output = solver.net.blobs['act_b_conv1'].data[0,0,:,:,:] * mask + visible
	output_single_class = solver_single_class.net.blobs['act_b_conv1'].data[0,0,:,:,:] * mask + visible
	output_3dshapenets = numpy.fromfile('../3DShapeNets/completed.data',dtype=numpy.float32).reshape(30,30,30).transpose()
	input_err = numpy.linalg.norm(complete_views.samples[id] - visible)**2 / 2
	output_err = numpy.linalg.norm(complete_views.samples[id] - output)**2 / 2
	output_single_class_err = numpy.linalg.norm(complete_views.samples[id] - output_single_class)**2 / 2
	output_3dshapenets_err = numpy.linalg.norm(complete_views.samples[id] - output_3dshapenets)**2 / 2
	fig = plt.figure()
	ax = fig.add_subplot(151, projection='3d')
	ax.set_xlim(0,30)
	ax.set_ylim(0,30)
	ax.set_zlim(0,30)
	x,y,z = numpy.nonzero(visible)
	color = src[x,y,z] * 2 - 1 
	ax.scatter(x,y,z,c=cm.jet(color),s=10)
	print 'Input error: '+str(input_err)
	ax = fig.add_subplot(152, projection='3d')
	ax.set_xlim(0,30)
	ax.set_ylim(0,30)
	ax.set_zlim(0,30)
	x,y,z = numpy.nonzero(output>0.1)
	color = output[x,y,z]
	ax.scatter(x,y,z,c=cm.jet(color),s=10)
	print 'Output error: '+str(output_err)
	ax = fig.add_subplot(153, projection='3d')
	ax.set_xlim(0,30)
	ax.set_ylim(0,30)
	ax.set_zlim(0,30)
	x,y,z = numpy.nonzero(output_single_class>0.1)
	color = output_single_class[x,y,z]
	ax.scatter(x,y,z,c=cm.jet(color),s=10)
	print 'Output (single class) error: '+str(output_single_class_err)
	ax = fig.add_subplot(154, projection='3d')
	ax.set_xlim(0,30)
	ax.set_ylim(0,30)
	ax.set_zlim(0,30)
	x,y,z = numpy.nonzero(output_3dshapenets>0.1)
	color = output_3dshapenets[x,y,z]
	ax.scatter(x,y,z,c=cm.jet(color),s=10)
	print 'Output (3dshapenets) error: '+str(output_3dshapenets_err)
	ax = fig.add_subplot(155, projection='3d')
	ax.set_xlim(0,30)
	ax.set_ylim(0,30)
	ax.set_zlim(0,30)
	x,y,z = numpy.nonzero(complete_views.samples[id])
	ax.scatter(x,y,z,c='r',s=10)
	print 'Completion sample '+classname+': index '+str(id)
	plt.show()
