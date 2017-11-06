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
f = h5py.File('Hands2017Train_32b.h5','r')

# write data to tfrecords file
writer = tf.python_io.TFRecordWriter("/home/dhri-dz/Hands2017Train_32")

for i in range(27032):
	# convert image and scale to bytes
	image = f['depth'][i].tobytes()
	joint = f['joint'][i].tobytes()
	handScale = f['handScale'][i].tobytes()
	fingerScale = f['fingerScale'][i].tobytes()
	center3ds = f['center3ds'][i].tobytes()
	center3dsOrig = f['center3dsOrig'][i].tobytes()
	offset = f['offset'][i].tobytes()

	# write to file
	example = tf.train.Example(features = tf.train.Features(feature={
		'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
		'joint': tf.train.Feature(bytes_list=tf.train.BytesList(value=[joint])),
		'handScale': tf.train.Feature(bytes_list=tf.train.BytesList(value=[handScale])),
		'fingerScale': tf.train.Feature(bytes_list=tf.train.BytesList(value=[fingerScale])),
		'center3ds': tf.train.Feature(bytes_list=tf.train.BytesList(value=[center3ds])),
		'center3dsOrig': tf.train.Feature(bytes_list=tf.train.BytesList(value=[center3dsOrig])),
		'offset': tf.train.Feature(bytes_list=tf.train.BytesList(value=[offset])),
		}))
	writer.write(example.SerializeToString())

writer.close()