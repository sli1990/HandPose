from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import math
import glob
import numpy as np
import h5py
import tensorflow as tf

# HDF5 file to convert
f = h5py.File('train_5.h5','r')

# write data to tfrecords file
writer = tf.python_io.TFRecordWriter("/home/dhri-dz/Documents/HandPose/Dataset/Training/HDF5_Test/HDF5TFR_5")

for i in range(30000):
	# convert image and scale to bytes
	binary_image = f['depth'][i].tobytes()
	scale = f['handScales'][i].tobytes()

	# write to file
	example = tf.train.Example(features = tf.train.Features(feature={
		'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[binary_image])),
		'scale': tf.train.Feature(bytes_list=tf.train.BytesList(value=[scale])),
		}))
	writer.write(example.SerializeToString())

writer.close()