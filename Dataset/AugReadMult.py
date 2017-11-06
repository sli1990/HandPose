from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import math
import numpy as np
from PIL import Image
import tensorflow as tf

reconstructed_images = []

record_iterator = tf.python_io.tf_record_iterator(path="/home/dhri-dz/Documents/HandPose/Augmentation/AugTest")

for string_record in record_iterator:
	parsed_features = tf.parse_single_example(string_record,features ={
		'image': tf.FixedLenFeature([],tf.string),
		'joint': tf.FixedLenFeature([],tf.string),
		'handScale': tf.FixedLenFeature([],tf.string),
		'fingerScale': tf.FixedLenFeature([],tf.string),
		'center3ds': tf.FixedLenFeature([],tf.string),
		'center3dsOrig': tf.FixedLenFeature([],tf.string),
		'offset': tf.FixedLenFeature([],tf.string),
		},name='features')

	# Decode content into correct types
	image_dec = tf.decode_raw(parsed_features['image'],tf.float32)
	joint_dec = tf.decode_raw(parsed_features['joint'],tf.float64)
	handScale_dec = tf.decode_raw(parsed_features['handScale'],tf.float64)
	fingerScale_dec = tf.decode_raw(parsed_features['fingerScale'],tf.float64)
	center3ds_dec = tf.decode_raw(parsed_features['center3ds'],tf.float64)
	center3dsOrig_dec = tf.decode_raw(parsed_features['center3dsOrig'],tf.float64)
	offset_dec = tf.decode_raw(parsed_features['offset'],tf.float64)

	# Reshape image to 192x192
	image_reshaped = tf.reshape(image_dec,[192,192,1])
	joint_dec = tf.cast(joint_dec, tf.float32)

	print(image_dec)
	print(joint_dec)
	print(handScale_dec)
	print(fingerScale_dec)
	print(center3ds_dec)
	print(center3dsOrig_dec)
	print(offset_dec)

	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)
	tf.train.start_queue_runners(sess=sess)
	print(sess.run(joint_dec))
	print(sess.run(handScale_dec))
	print(sess.run(fingerScale_dec))
	print(sess.run(center3ds_dec))
	print(sess.run(center3dsOrig_dec))
	print(sess.run(offset_dec))
	sess.close()