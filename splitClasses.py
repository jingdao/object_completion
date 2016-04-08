import sys
import time
import caffe
from importViews import *

partial_views = GridData('data/partial_view.data','data/labels.data')
complete_views = GridData('data/complete_view.data','data/labels.data')
features = numpy.load('data/features.npy')

partial_raw = open('data/partial_view.data','rb')
complete_raw = open('data/complete_view.data','rb')

fd={}
size=partial_views.data_size ** 3
for i in range(partial_views.num_samples):
	l = partial_views.labels[i]
	lb = partial_views.label_names[i]
	if not lb in fd:
		print lb
		fd[lb] = []
		fd[lb].append(open('data/'+lb+'_partial.data','wb'))
		fd[lb].append(open('data/'+lb+'_complete.data','wb'))
		fd[lb].append(open('data/'+lb+'_labels.data','wb'))
		fd[lb].append(None)
	fd[lb][0].write(partial_raw.read(size))
	fd[lb][1].write(complete_raw.read(size))
	fd[lb][2].write(str(l)+' '+lb+'\n')
	if fd[lb][3] is None:
		fd[lb][3] = numpy.copy(features[i,:])
	else:
		fd[lb][3] = numpy.vstack((fd[lb][3],features[i,:]))

partial_raw.close()
complete_raw.close()
for lb in fd:
	fd[lb][0].close()
	fd[lb][1].close()
	fd[lb][2].close()
	numpy.save('data/'+lb+'_features.npy',fd[lb][3])
