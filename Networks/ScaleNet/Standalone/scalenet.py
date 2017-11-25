# Python 2 Compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports 
import math
import tensorflow as tf
import numpy as np
from layers import *
from dataprocessing import *


def ScaleNet(images,joints,scales):
	"""ScaleNet model
  	Args:	
  		images: image batch to be processed
  	Returns:
    	box_offset: offset to shift COM bounding box by
  	"""

  	# Placeholders
  	#keep_prob = tf.placeholder(tf.float32)

	# Scope vairables for ScaleNet
  	with tf.name_scope('scalenet'):

	  	# Padding Layer (with value 1) to maintain size
	  	with tf.name_scope('padd'):
	  		padded_input = PADD2D(images, 2, 1)

	  	# Convolutional Layer 1
	  	with tf.name_scope('convolution1'):
	  		hidden1 = CONV2D(padded_input, 5, 1, 12, 'VALID', 0)

	  	# Pooling Layer 1
	  	with tf.name_scope('pool1'):
	  		hidden1_pool = MAXPOOL2D(hidden1, 4, 4, 'SAME')

	  	# Padding Layer (with value 1) to maintain size
	  	with tf.name_scope('padd2'):
	  		padded_2 = PADD2D(hidden1_pool, 2, 1)

	  	# Convolutional Layer 2
	  	with tf.name_scope('convolution2'):
	  		hidden2 = CONV2D(padded_2, 5, 12, 12, 'VALID', 0)

	  	# Pooling Layer 2
	  	with tf.name_scope('pool2'):
	  		hidden2_pool = MAXPOOL2D(hidden2, 2, 2, 'SAME')

	  	# Padding Layer (with value 1) to maintain size
	  	with tf.name_scope('padd3'):
	  		padded_3 = PADD2D(hidden2_pool, 1, 1)

	  	# Convolutional Layer 3
	  	with tf.name_scope('convolution3'):
	  		hidden3 = CONV2D(padded_3, 3, 12, 12, 'VALID', 0)

	  	# Flatten output from convolutional layers
	  	with tf.name_scope('flatten'):
	  		hidden3_flat = RESHAPE2D(hidden3, 16, 12)

	  	# Fully Connected Layer 1
	  	with tf.name_scope('fc1'):
	  		hidden4 = FULLY_CONNECTED(hidden3_flat, 16*16*12, 1024, 0, False, 0.0)

	  	# Fully Connected Layer 2
	  	with tf.name_scope('fc2'):
	  		hidden5 = FULLY_CONNECTED(hidden4, 1024, 1024, 0, False, 0.0)

	  	# Linear Output Layer
	  	with tf.name_scope('linear'):
	  		hand_scale = LINEAR_OUTPUT(hidden5, 1024, 1, 0)
  	
  	return hand_scale


def loss_scalenet(hscale, label):
	"""Calculates the loss from the hand scale and the label.
	Args:
		hscale: Hand scale tensor, float - [batch_size].
		label: Label tensor, float - [batch_size].
	Returns:
		loss: Loss tensor of type float.
	"""
	hscale_error = hscale - label
	l2_hscale_loss = tf.nn.l2_loss(hscale_error, name='l2_hscale')
	
	return l2_hscale_loss


def train_scalenet(loss, learning_rate):
	"""Sets up the training operations.
	Creates a summarizer to track the loss and the learning rate over time in TensorBoard.
	Creates an optimizer and applies the gradients to all trainable variables.
	The operation returned by this function is what must be passed to the sess.run() call to cause the model to train.
	Args:
		loss: Loss tensor, from loss().
		learning_rate: The learning rate to be used for Adam.
	Returns:
		train_op: The Op for training.
	"""
	
	# Add a scalar summary for the snapshot loss.
	tf.summary.scalar('loss', loss)
	tf.summary.scalar('learning_rate', learning_rate)
	
	# Create the Adam optimizer with the given learning rate.
	optimizer = tf.train.AdamOptimizer(learning_rate)
	
	# Create a variable to track the global step.
	global_step = tf.Variable(0, name='global_step', trainable=False)
	
	# Use the optimizer to apply the gradients that minimize the loss (and also increment the global step counter) as a single training step.
	train_op = optimizer.minimize(loss, global_step=global_step)
	
	return train_op
