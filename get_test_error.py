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
use_testset = True
if use_testset:
	data_folder = 'data/test_separate_class_correct_view/'
else:
	data_folder = 'data/separate_class_correct_view/'
solver.net.blobs['data'].reshape(1,1,30,30,30)
solver.net.blobs['label'].reshape(1,1,30,30,30)
if use_mask:
	solver.net.blobs['visible'].reshape(1,1,30,30,30)
	solver.net.blobs['mask'].reshape(1,1,30,30,30)

classes = ['bathtub','bed','chair','desk','dresser','monitor','night_stand','sofa','table','toilet']

overall_error = []
comp_time = []
for classname in classes:
	partial_views = GridData(data_folder+classname+'_partial.data',data_folder+classname+'_labels.data')
	complete_views = GridData(data_folder+classname+'_complete.data',data_folder+classname+'_labels.data')
	
	class_samples = partial_views.num_samples
	class_error = numpy.zeros(class_samples)
	for i in range(class_samples):
		src = initialize_missing(partial_views.samples[i])
		visible = numpy.array(src == 1,dtype=numpy.float)
		mask = numpy.array(src == 0.5,dtype=numpy.float)
		solver.net.blobs['data'].data[0,0,:,:,:] = src
		solver.net.blobs['label'].data[0,0,:,:,:] = complete_views.samples[i]
		if use_mask:
			solver.net.blobs['visible'].data[0,0,:,:,:] = visible
			solver.net.blobs['mask'].data[0,0,:,:,:] = mask
		start = time.clock()
		solver.net.forward()
		end = time.clock()
		comp_time.append(end-start)
		output = solver.net.blobs['act_b_conv1'].data[0,0,:,:,:] * mask + visible
		#output = solver.net.blobs['act_b_conv1'].data[0,0,:,:,:]
		class_error[i] = numpy.linalg.norm(complete_views.samples[i] - output)**2 / 2
		#class_error[i] = float(solver.net.blobs['loss'].data)
	numpy.save('results/'+classname+'_test_error',class_error)
	min_err = numpy.min(class_error)
	min_id = numpy.argmin(class_error)
	max_err = numpy.max(class_error)
	max_id = numpy.argmax(class_error)
	mean_err = numpy.mean(class_error)
	err25 = numpy.percentile(class_error,25)
	err50 = numpy.percentile(class_error,50)
	err75 = numpy.percentile(class_error,75)
	overall_error.extend(class_error)
	print '%s,%.0f,%.0f,%.0f,%.0f,%.0f' % (classname,min_err,err25,err50,err75,max_err)

min_err = numpy.min(overall_error)
max_err = numpy.max(overall_error)
err25 = numpy.percentile(overall_error,25)
err50 = numpy.percentile(overall_error,50)
err75 = numpy.percentile(overall_error,75)
overall_error.extend(overall_error)
print 'overall,%.0f,%.0f,%.0f,%.0f,%.0f' % (min_err,err25,err50,err75,max_err)
print 'completion time: %f avg (%d samples)' % (numpy.mean(comp_time),len(comp_time))
