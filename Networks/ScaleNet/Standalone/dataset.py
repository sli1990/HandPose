# Python 2 Compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports 
import math
import tensorflow as tf
from dataprocessing import *


def dataset_filenames(data_dir,mode):

	if mode=='CHALLENGE':

		# Training and validation data locations
		training_filenames = ["Hands2017Train176_1",
							  "Hands2017Train176_2", 
							  "Hands2017Train176_3",
							  "Hands2017Train176_4",
							  "Hands2017Train176_5",
							  "Hands2017Train176_6",
							  "Hands2017Train176_7",
							  "Hands2017Train176_8",
							  "Hands2017Train176_9",
							  "Hands2017Train176_10",
							  "Hands2017Train176_11",
							  "Hands2017Train176_12",
							  "Hands2017Train176_13",
							  "Hands2017Train176_14",
							  "Hands2017Train176_15",
							  "Hands2017Train176_16",
							  "Hands2017Train176_17",
							  "Hands2017Train176_18",
							  "Hands2017Train176_19",
							  "Hands2017Train176_20",
							  "Hands2017Train176_21",
							  "Hands2017Train176_22",
							  "Hands2017Train176_23",
							  "Hands2017Train176_24",
							  "Hands2017Train176_25",
							  "Hands2017Train176_26",
							  "Hands2017Train176_27",
							  "Hands2017Train176_28",
							  "Hands2017Train176_29",
							  "Hands2017Train176_30"]
							  
	  	validation_filenames = ["Hands2017Train176_31",
	  	 						"Hands2017Train176_32"]
	  	
	elif mode=='TEST':

		# Training and validation data locations
		training_filenames = ["Hands2017Train176_1"]
							  
	  	validation_filenames = ["Hands2017Train176_2"]

	elif mode=='TEST128C':

		# Training and validation data locations !!! COMMENT OUT OFFSET WHEN PARSING AS NOT IN DATASET
		training_filenames = ["Hands2017Train128_1"]
							  
	  	validation_filenames = ["Hands2017Train128_2"]

	# Append directory where data is located
	training_filenames = [data_dir+s for s in training_filenames]
	validation_filenames = [data_dir+s for s in validation_filenames]

	return training_filenames, validation_filenames


def parse_function(example_proto):
	"""Parse through current binary batch and extract images and labels"""

	# Parse through features and extract byte string
	parsed_features = tf.parse_single_example(example_proto,features ={
		'image': tf.FixedLenFeature([],tf.string),
		'joint': tf.FixedLenFeature([],tf.string),
		#'offset': tf.FixedLenFeature([],tf.string),
		'handScale': tf.FixedLenFeature([],tf.string)
		},name='features')

	# Decode content into correct types
	image_dec = tf.decode_raw(parsed_features['image'],tf.float32)
	joint_dec = tf.decode_raw(parsed_features['joint'],tf.float32)
	#offset_dec = tf.decode_raw(parsed_features['offset'],tf.float32)
	handScale_dec = tf.decode_raw(parsed_features['handScale'],tf.float32)

	# Reshape image to 128x128
	image_reshaped = tf.reshape(image_dec,[128,128,1])

	# IN CASE 176x176 DATASET IS USED: SIMULATE PERFECT BOXNET BY USING GROUND TRUTH
	#image_reshaped, joint_dec = process_box2scale(image_reshaped,joint_dec,offset_dec)

	return image_reshaped, joint_dec, handScale_dec


def parse_function_augment(example_proto):
	"""Parse through current binary batch and extract images and labels"""

	# Parse through features and extract byte string
	parsed_features = tf.parse_single_example(example_proto,features ={
		'image': tf.FixedLenFeature([],tf.string),
		'joint': tf.FixedLenFeature([],tf.string),
		#'offset': tf.FixedLenFeature([],tf.string),
		'handScale': tf.FixedLenFeature([],tf.string)
		},name='features')

	# Decode content into correct types
	image_dec = tf.decode_raw(parsed_features['image'],tf.float32)
	joint_dec = tf.decode_raw(parsed_features['joint'],tf.float32)
	#offset_dec = tf.decode_raw(parsed_features['offset'],tf.float32)
	handScale_dec = tf.decode_raw(parsed_features['handScale'],tf.float32)

	# Reshape image to 128x128
	image_reshaped = tf.reshape(image_dec,[128,128,1])

	# IN CASE 176x176 DATASET IS USED: SIMULATE PERFECT BOXNET BY USING GROUND TRUTH
	#image_reshaped, joint_dec = process_box2scale(image_reshaped,joint_dec,offset_dec)

	# Data Augmentation
	image_reshaped, joint_dec, handScale_dec = tf.py_func(augmentation_cv,[image_reshaped, joint_dec, handScale_dec],[tf.float32, tf.float32, tf.float32, tf.float32])
	image_reshaped = tf.reshape(image_reshaped,[128,128,1])

	# TF IMPLEMENTATION OF DATA AUGMENTATION: MIGHT BE SLOWER WHEN TF IS NOT COMPILED FROM SOURCE
	# image_reshaped, joint_dec, handScale_dec = augmentation(image_reshaped, joint_dec, handScale_dec)

	return image_reshaped, joint_dec, handScale_dec


def create_datasets_boxnet(training_filenames, validation_filenames, handle, batch_size, thread_count, buffer_count):

  	# Define the training with Dataset API
	training_dataset = tf.contrib.data.TFRecordDataset(training_filenames)
	training_dataset = training_dataset.map(parse_function_augment, num_threads=thread_count)
	training_dataset = training_dataset.shuffle(buffer_size=buffer_count)
	training_dataset = training_dataset.batch(batch_size)	
	training_dataset = training_dataset.repeat()

	# Define the evaluation on training dataset dataset
	eval_train_dataset = tf.contrib.data.TFRecordDataset(training_filenames)
	eval_train_dataset = eval_train_dataset.map(parse_function, num_threads=thread_count)
	eval_train_dataset = eval_train_dataset.repeat(2)
	eval_train_dataset = eval_train_dataset.batch(batch_size)

	# Define the validation dataset for evaluation
	validation_dataset = tf.contrib.data.TFRecordDataset(validation_filenames)
	validation_dataset = validation_dataset.map(parse_function, num_threads=thread_count)
	validation_dataset = validation_dataset.repeat(2)
	validation_dataset = validation_dataset.batch(batch_size)	
	
	# Create a feedable iterator to consume data
	iterator = tf.contrib.data.Iterator.from_string_handle(handle, training_dataset.output_types, training_dataset.output_shapes)
	next_images, next_joints, next_scales = iterator.get_next()

	# Define the different iterators
	training_iterator = training_dataset.make_one_shot_iterator()
	eval_train_iterator = eval_train_dataset.make_initializable_iterator()
	validation_iterator = validation_dataset.make_initializable_iterator()

	return next_images, next_joints, next_scales, training_iterator, eval_train_iterator, validation_iterator