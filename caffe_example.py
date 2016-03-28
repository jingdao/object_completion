import caffe
caffe.set_mode_cpu()
from caffe import layers as L, params as P


solver = caffe.SGDSolver('3Dshapenet_solver.prototxt')
# assign data
"""
batchsize = 10;
solver.net.blobs['data'].reshape(batchsize,1,30,30,30)
solver.net.blobs['data'].data[...] = data
solver.net.blobs['label'].reshape(batchsize)
solver.net.blobs['label'].data[...] = label
"""
# initialize wights and bias
# Load parameters
"""
solver.net.params['conv1'][0].data[...] =  #weigth
solver.net.params['conv1'][1].data[...] =  #bias
solver.net.params['conv2'][0].data[...] =  #weigth
solver.net.params['conv2'][1].data[...] =  #bias
solver.net.params['conv3'][0].data[...] =  #weigth
solver.net.params['conv3'][1].data[...] =  #bias
"""

solver.net.forward()
