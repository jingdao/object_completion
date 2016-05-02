Instructions
=============

Dependencies
------------

The project runs on a framework of Caffe and Python. The caffe library can be downloaded from
https://github.com/BVLC/caffe. Follow the instructios in the repository to install dependencies for Caffe.

Data files
----------
The data files consist of preprocessed 30x30x30 voxel grids from the ModelNet10 database. 
(http://modelnet.cs.princeton.edu/)
First download the following files to the data/ folder
https://drive.google.com/open?id=0B7cRYqpXc0QJMUx1VE0xSVM2MU0 (partial_view.data)
https://drive.google.com/open?id=0B7cRYqpXc0QJaVJGYUpzcDJGdkk (complete_view.data)
https://drive.google.com/open?id=0B7cRYqpXc0QJNGxTS3JhSkFWX1E (labels.data)

Training stage
--------------
To train sample completion on all 10 classes run:

	python sample_completion.py architecture/1200/Net3DReg_solver.prototxt 

To train sample completion with output mask run:
	
	python sample_completion_difference.py architecture/1200-difference/Net3DReg_solver.prototxt

The trained network is saved to the results folder in the form of caffemodel and solverstate files.
	
Testing stage
-------------

The test error for a specified architecture can be obtained by:

	python get_test_error.py architecture/1200/Net3DReg_solver.prototxt results/1200/_iter_100000.caffemodel
	
Generate Completions
--------------------

Completed views for objects can be generated for a specified architecture by:

	python generate_completion.py architecture/1200/Net3DReg_solver.prototxt results/1200/_iter_100000.caffemodel results/single/1200/_iter_100000.caffemodel
