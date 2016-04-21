import sys
import time
import caffe
import os
if os.system('nvcc --version') == 0:
	caffe.set_mode_gpu()
else:
	caffe.set_mode_cpu()
from importViews import *

def initialize_missing(data):
	mask = data < 0
	filler = numpy.ones(len(data[mask])) * 0.5
	res = numpy.array(data,dtype=numpy.float32)
	res[mask] = filler
	return res

solver_path = sys.argv[1]
solver = caffe.AdamSolver(solver_path)
path_net = sys.argv[2]
solver.net.copy_from(path_net)
use_mask = 'mask' in solver.net.blobs.keys()
solver.net.blobs['data'].reshape(1,1,30,30,30)
solver.net.blobs['label'].reshape(1,1,30,30,30)
if use_mask:
	solver.net.blobs['visible'].reshape(1,1,30,30,30)
	solver.net.blobs['mask'].reshape(1,1,30,30,30)

classes = ['bathtub','bed','chair','desk','dresser','monitor','night_stand','sofa','table','toilet']

avg_error = 0
for classname in classes:
	partial_views = GridData('data/test_separate_class_correct_view/'+classname+'_partial.data','data/test_separate_class_correct_view/'+classname+'_labels.data')
	complete_views = GridData('data/test_separate_class_correct_view/'+classname+'_complete.data','data/test_separate_class_correct_view/'+classname+'_labels.data')
	
	class_samples = partial_views.num_samples
	class_error = numpy.zeros(class_samples)
	for i in range(class_samples):
		src = initialize_missing(partial_views.samples[i])
		solver.net.blobs['data'].data[0,0,:,:,:] = src
		solver.net.blobs['label'].data[0,0,:,:,:] = complete_views.samples[i]
		if use_mask:
			solver.net.blobs['visible'].data[0,0,:,:,:] = numpy.array(src == 1,dtype=numpy.float)
			solver.net.blobs['mask'].data[0,0,:,:,:] = numpy.array(src == 0.5,dtype=numpy.float)
		solver.net.forward()
		class_error[i] = float(solver.net.blobs['loss'].data)
	min_err = numpy.min(class_error)
	min_id = numpy.argmin(class_error)
	max_err = numpy.max(class_error)
	max_id = numpy.argmax(class_error)
	mean_err = numpy.mean(class_error)
	avg_error += mean_err
	print '%s(%d): %.0f(%d) min %.0f(%d) max %.0f avg' % (classname,class_samples,min_err,min_id,max_err,max_id,mean_err)

print 'Average: '+str(avg_error / len(classes))
