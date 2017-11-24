# Python 2 Compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports 
import math
import time
import tensorflow as tf


def evaluation_function(sess, eval_op, initializable_iterator, handle, data_handle, data_size, steps_per_epoch, mode):

	# Number of regressed parameters
	reg_param = 3

	# Start timer
	start_time = time.time()
			
	# Initialize iterator
	sess.run(initializable_iterator.initializer)

	# Initialize loss counter
	eval_loss = 0		

	# Iterate over training dataset
	for _ in range(steps_per_epoch):
		eval_loss += sess.run(eval_op, feed_dict={handle: data_handle})

	# Calculate average training data evaluation loss
	eval_loss_avg = eval_loss/(data_size*reg_param)
	eval_loss_avg_mm = 1000*math.sqrt(2*eval_loss_avg)

	# Print training data evaluation results
	duration = time.time() - start_time
	if mode=='TRAINING':
		print('TRAIN DATA EVALUATION: Num examples: %d  Avg loss per offset component: %.8f (%.2f mm) Time for evaluation: %.2f sec' % (data_size, eval_loss_avg, eval_loss_avg_mm, duration))
	elif mode=='VALIDATION':
		print('VALIDATION DATA EVALUATION: Num examples: %d  Avg loss per offset component: %.8f (%.2f mm) Time for evaluation: %.2f sec' % (data_size, eval_loss_avg, eval_loss_avg_mm, duration))
			

def inference_fun(sess, eval_op, eval_train_iterator, validation_iterator, handle, eval_train_handle, validation_handle, train_size, validation_size, eval_train_steps_per_epoch, validation_steps_per_epoch):

	# EVALUATION RUN ON TRAINING SET
	evaluation_function(sess, eval_op, eval_train_iterator, handle, eval_train_handle, train_size, eval_train_steps_per_epoch, 'TRAINING')

	# EVALUATION RUN ON VALIDATION SET
	evaluation_function(sess, eval_op, validation_iterator, handle, validation_handle, validation_size, validation_steps_per_epoch, 'VALIDATION')