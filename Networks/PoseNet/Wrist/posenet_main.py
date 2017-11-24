# Python 2 Compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import argparse
import os
import sys
import time
import math
from six.moves import xrange 

import tensorflow as tf
from posenet import *
from evaluation import *
from dataset import *

# Basic model parameters as external flags.
FLAGS = None


def run_posenet():
	"""Train the PoseNet on the hand image data."""

	# Create the iterators and data placeholders for the datasets
	training_filenames, validation_filenames = dataset_filenames(FLAGS.data_dir, 'TEST')
	handle = tf.placeholder(tf.string, shape=[])
	next_images, next_joints, next_scales, training_iterator, eval_train_iterator, validation_iterator = create_datasets_boxnet(training_filenames, validation_filenames, handle, FLAGS.batch_size, 4, 1000)

	# Build a Graph that computes predictions from the PoseNet inference model
	pose3D_est, wristBase_est, fingerVectors_est, jointAngles_est, boneLengths_est, poseDirect_est = PoseNet(next_images, next_joints, next_scales)

	# Add the operations for loss calculation to the graph
	eval_op = loss_boxnet(pose3D_est, wristBase_est, fingerVectors_est, jointAngles_est, boneLengths_est, poseDirect_est, next_joints)

	# Add the operation that calculates and applies the gradients to the graph
	train_op = train_boxnet(eval_op, FLAGS.learning_rate)

	# Build the summary Tensor based on the TF collection of Summaries.
	merged = tf.summary.merge_all()

	# Initialize the global variables
	init = tf.global_variables_initializer()

	# Create a saver for writing training checkpoints (the 10 most recent are kept).
	saver = tf.train.Saver(max_to_keep=10)

	# Create a session for running operations on the Graph.
	sess = tf.Session()
	#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

	# Instantiate SummaryWriters to output summaries and the Graph.
	train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
	validation_writer = tf.summary.FileWriter(FLAGS.log_dir + '/eval')

	# number of steps per epoch
	eval_train_steps_per_epoch = int(math.ceil(FLAGS.train_size/FLAGS.batch_size))
	validation_steps_per_epoch = int(math.ceil(FLAGS.validation_size/FLAGS.batch_size))

	# Create the needed handles to feed the handle placeholder
	training_handle = sess.run(training_iterator.string_handle())
	eval_train_handle = sess.run(eval_train_iterator.string_handle())
	validation_handle = sess.run(validation_iterator.string_handle())


	# Select mode to run network in
	if FLAGS.run_mode=='inference':

		# RESTORE THE MODEL
		checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt-2728124')
		saver.restore(sess, checkpoint_file)

		# START INFERENCE
		print('START INFERENCE')
		inference_fun(sess, eval_op, eval_train_iterator, validation_iterator, handle, eval_train_handle, validation_handle, FLAGS.train_size, FLAGS.validation_size, eval_train_steps_per_epoch, validation_steps_per_epoch)

	else:
		
		# Some control variables
		log_frequency_comp = FLAGS.log_frequency - 1
		eval_frequency = int(math.ceil(FLAGS.train_size/FLAGS.batch_size))
		eval_frequency_comp = eval_frequency - 1

		if FLAGS.run_mode=='training':

			# Run the Op to initialize the variables.
			sess.run(init)

		elif FLAGS.run_mode=='cont_training':
			
			# Restore the model
			checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt-5627')
			saver.restore(sess, checkpoint_file)

			# START INFERENCE (confirm that correct model has been loaded)
			print('START INFERENCE')
			inference_fun(sess, eval_op, eval_train_iterator, validation_iterator, handle, eval_train_handle, validation_handle, FLAGS.train_size, FLAGS.validation_size, eval_train_steps_per_epoch, validation_steps_per_epoch)
		

		# Start the training loop.
		print('START TRAINING')
		start_time = time.time()

		for steps in xrange(FLAGS.start_step ,FLAGS.max_steps):
			
			# TRAINING
			if steps % FLAGS.log_frequency == log_frequency_comp:
				# Logging for Tensorboard
				#run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
				#run_metadata = tf.RunMetadata()
				summary, _ = sess.run([merged, train_op], feed_dict={handle: training_handle})#, options=run_options, run_metadata=run_metadata)
				#train_writer.add_run_metadata(run_metadata,'step%d' % steps)
				train_writer.add_summary(summary,steps)
				train_writer.flush()
			else:
				sess.run(train_op, feed_dict={handle: training_handle})

			# EVALUATION OF TRAINING
			if steps % eval_frequency == eval_frequency_comp:

				# END OF EPOCH
				duration = time.time() - start_time
				print('Step %d completed. Duration of last %d steps: %.2f sec' % (steps+1, eval_frequency, duration))

				# EVALUATION RUN
				inference_fun(sess, eval_op, eval_train_iterator, validation_iterator, handle, eval_train_handle, validation_handle, FLAGS.train_size, FLAGS.validation_size, eval_train_steps_per_epoch, validation_steps_per_epoch)

				# SAVE A MODEL CHECKPOINT
				checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
				saver.save(sess, checkpoint_file, global_step=steps)

				# RESET TIMER
				start_time = time.time()


def main(_):
	if FLAGS.run_mode=='training':
		if tf.gfile.Exists(FLAGS.log_dir):
			tf.gfile.DeleteRecursively(FLAGS.log_dir)
		tf.gfile.MakeDirs(FLAGS.log_dir)
	run_posenet()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
	  '--learning_rate',
	  type=float,
	  default=0.0001,
	  help='Initial learning rate.'
  )
  parser.add_argument(
	  '--batch_size',
	  type=int,
	  default=32,
	  help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
	  '--start_step',
	  type=int,
	  default=0,
	  help='Training step to start with.'
  )
  parser.add_argument(
	  '--max_steps',
	  type=int,
	  default=93800,
	  help='Number of steps to run trainer.'
  )
  parser.add_argument(
	  '--data_dir',
	  type=str,
	  default='/home/dhri-dz/',
	  help='Directory where the dataset lies.'
  )
  parser.add_argument(
	  '--train_size',
	  type=int,
	  default=30000,
	  help='Training data size (number of images).'
  )
  parser.add_argument(
	  '--validation_size',
	  type=int,
	  default=30000,
	  help='Validation data size (number of images).'
  )
  parser.add_argument(
	  '--log_dir',
	  type=str,
	  default='/home/dhri-dz/Documents/HandPose/T176/Log',
	  help='Directory to put the log data.'
  )
  parser.add_argument(
	  '--log_frequency',
	  type=int,
	  default=1000,
	  help='Frequency (steps) with which data is logged'
  )
  parser.add_argument(
	  '--run_mode',
	  type=str,
	  default='training',
	  help='Mode in which network is run: training, cont_training, inference'
  )

  FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)