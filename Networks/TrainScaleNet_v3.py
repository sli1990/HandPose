from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import argparse
import os
import sys
import time
import math

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# MAKROS
IMAGE_SIZE = 128
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

# Basic model parameters as external flags.
FLAGS = None


def lrelu(x, leak=0.01, name="lrelu"):
	# leaky ReLU unit
  	with tf.variable_scope('lrelu_def'):
  		f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

         
def scalenet(images):
	"""ScaleNet model to be used for inference
  	Args:	
  		images: image batch to be processed
  	Returns:
    	scale: hand scale parameter
  	"""

  	# Padding Layer (with value 1) to maintain size
  	with tf.name_scope('padd'):
  		padded_input = tf.pad(images,[[0, 0],[2, 2],[2, 2],[0, 0]],"CONSTANT",constant_values=1)

  	# Convolutional Layer 1 <-different learning rate for biases than for weights to be implemented
  	with tf.name_scope('convolution1'):
		weights = tf.Variable(tf.random_uniform([5, 5, 1, 12], minval=-math.sqrt(3)/IMAGE_PIXELS, maxval=math.sqrt(3)/IMAGE_PIXELS), name='weights')
		biases = tf.Variable(tf.zeros([12]), name='biases')
		hidden1 = tf.nn.relu(tf.nn.conv2d(padded_input, weights, strides=[1, 1, 1, 1], padding='VALID') + biases)
	#hidden1 = lrelu(tf.nn.conv2d(padded_input, weights, strides=[1, 1, 1, 1], padding='VALID') + biases)

  	# Pooling Layer 1
  	with tf.name_scope('pool1'):
  		hidden1_pool = tf.nn.max_pool(hidden1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

  	# Convolutional Layer 2 <-different learning rate for biases than for weights to be implemented
  	with tf.name_scope('convolution2'):
  		weights = tf.Variable(tf.random_uniform([5, 5, 12, 12], minval=-math.sqrt(3)/(12*IMAGE_PIXELS/16), maxval=math.sqrt(3)/(12*IMAGE_PIXELS/16)), name='weights')
  		biases = tf.Variable(tf.zeros([12]), name='biases')
  		hidden2 = tf.nn.relu(tf.nn.conv2d(hidden1_pool, weights, strides=[1, 1, 1, 1], padding='SAME') + biases)
  		#hidden2 = lrelu(tf.nn.conv2d(hidden1_pool, weights, strides=[1, 1, 1, 1], padding='SAME') + biases)

  	# Pooling Layer 2
  	with tf.name_scope('pool2'):
  		hidden2_pool = tf.nn.max_pool(hidden2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  	# Convolutional Layer 3 <-different learning rate for biases than for weights to be implemented
  	with tf.name_scope('convolution3'):
  		weights = tf.Variable(tf.random_uniform([3, 3, 12, 12], minval=-math.sqrt(3)/(12*IMAGE_PIXELS/64), maxval=math.sqrt(3)/(12*IMAGE_PIXELS/64)), name='weights')
  		biases = tf.Variable(tf.zeros([12]), name='biases')
  		hidden3 = tf.nn.relu(tf.nn.conv2d(hidden2_pool, weights, strides=[1, 1, 1, 1], padding='SAME') + biases)
  		#hidden3 = lrelu(tf.nn.conv2d(hidden2_pool, weights, strides=[1, 1, 1, 1], padding='SAME') + biases)

  	# Flatten output from convolutional layers
  	with tf.name_scope('flatten'):
  		hidden3_flat = tf.reshape(hidden3, [-1, 16*16*12])

  	# Fully Connected Layer 1 <-different learning rate for biases than for weights to be implemented
  	with tf.name_scope('fc1'):
  		weights = tf.Variable(tf.random_uniform([16*16*12,1024], minval=-math.sqrt(3)/(12*IMAGE_PIXELS/64), maxval=math.sqrt(3)/(12*IMAGE_PIXELS/64)), name='weights')
  		biases = tf.Variable(tf.zeros([1024]), name='biases')
  		hidden4 = tf.nn.relu(tf.matmul(hidden3_flat, weights) + biases)
  		#hidden4 = lrelu(tf.matmul(hidden3_flat, weights) + biases)

  	# Fully Connected Layer 2 <-different learning rate for biases than for weights to be implemented
  	with tf.name_scope('fc2'):
  		weights = tf.Variable(tf.random_uniform([1024,1024], minval=-math.sqrt(3)/1024, maxval=math.sqrt(3)/1024), name='weights')
  		biases = tf.Variable(tf.zeros([1024]), name='biases')
  		hidden5 = tf.nn.relu(tf.matmul(hidden4, weights) + biases)
  		#hidden5 = lrelu(tf.matmul(hidden4, weights) + biases)

  	# Output Layer <-different learning rate for biases than for weights to be implemented
  	with tf.name_scope('linear'):
  		weights = tf.Variable(tf.random_uniform([1024,1], minval=-math.sqrt(3)/1024, maxval=math.sqrt(3)/1024), name='weights')
  		biases = tf.Variable(tf.zeros([1]), name='biases')
  		scale = tf.nn.relu(tf.matmul(hidden5, weights) + biases) 
  		#scale = lrelu(tf.matmul(hidden5, weights) + biases) 
  	
  	return scale

def loss(scale, label):
	"""Calculates the loss from the estimated scale and the label.
	Args:
		scale: Scale tensor, float - [batch_size].
		label: Label tensor, float - [batch_size].
	Returns:
		loss: Loss tensor of type float.
	"""
	scale_error = scale - label
	l2_scale_loss = tf.nn.l2_loss(scale_error, name='l2_scale')
	
	return tf.reduce_mean(l2_scale_loss, name='l2_scale_mean')


def training(loss, learning_rate):
	"""Sets up the training operations.
	Creates a summarizer to track the loss over time in TensorBoard.
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


def evaluation(scale, label):
	"""Evaluate the quality of predicting the correct scale on chosen dataset.
	Args:
		scale: Scale tensor, float - [batch_size].
		label: Label tensor, float - [batch_size].
	Returns:
		Loss tensor of type float.
	"""
	scale_error = scale - label
	l2_scale_loss = tf.nn.l2_loss(scale_error, name='l2_scale')
	
	return tf.reduce_mean(l2_scale_loss, name='l2_scale_mean')


def parse_function(example_proto):
	"""Parse through current binary batch and extract images and labels"""
	# Parse through features and extract byte string
	parsed_features = tf.parse_single_example(example_proto,features ={
		'image': tf.FixedLenFeature([],tf.string),
		'scale': tf.FixedLenFeature([],tf.string)
		},name='features')

	# Decode content into correct types
	image_dec = tf.decode_raw(parsed_features['image'],tf.float32)
	scale_dec = tf.decode_raw(parsed_features['scale'],tf.float32)

	# Reshape image to 128x128
	image_reshaped = tf.reshape(image_dec,[128,128,1])

	return image_reshaped, scale_dec


def run_training():
	"""Train the ScaleNet on the hand image data."""

	# Training and validation data locations
	training_filenames = ["/home/dhri-dz/Documents/HandPose/Dataset/Training/HDF5_Test/HDF5TFR_1", "/home/dhri-dz/Documents/HandPose/Dataset/Training/HDF5_Test/HDF5TFR_2", "/home/dhri-dz/Documents/HandPose/Dataset/Training/HDF5_Test/HDF5TFR_3", "/home/dhri-dz/Documents/HandPose/Dataset/Training/HDF5_Test/HDF5TFR_4"]
	validation_filenames = ["/home/dhri-dz/Documents/HandPose/Dataset/Training/HDF5_Test/HDF5TFR_5"]
  	
  	# Define the training with Dataset API
	training_dataset = tf.contrib.data.TFRecordDataset(training_filenames)
	training_dataset = training_dataset.map(parse_function)
	training_dataset = training_dataset.shuffle(buffer_size=1000)
	training_dataset = training_dataset.batch(FLAGS.batch_size)	
	training_dataset = training_dataset.repeat()

	# Define the evaluation on training dataset dataset
	eval_train_dataset = tf.contrib.data.TFRecordDataset(training_filenames)
	eval_train_dataset = eval_train_dataset.map(parse_function)
	eval_train_dataset = eval_train_dataset.batch(FLAGS.batch_size)

	# Define the validation dataset for evaluation
	validation_dataset = tf.contrib.data.TFRecordDataset(validation_filenames)
	validation_dataset = validation_dataset.map(parse_function)
	validation_dataset = validation_dataset.batch(FLAGS.batch_size)	
	
	# Create a feedable iterator to consume data
	handle = tf.placeholder(tf.string, shape=[])
	iterator = tf.contrib.data.Iterator.from_string_handle(handle, training_dataset.output_types, training_dataset.output_shapes)
	next_images, next_labels = iterator.get_next()

	# Define the different iterators
	training_iterator = training_dataset.make_one_shot_iterator()
	eval_train_iterator = eval_train_dataset.make_initializable_iterator()
	validation_iterator = validation_dataset.make_initializable_iterator()

	# Build a Graph that computes predictions from the ScaleNet inference model
	scales_est = scalenet(next_images)

    # Add the operations for loss calculation to the graph
	loss_val = loss(scales_est, next_labels)
	#tf.summary.scalar('loss_val',loss_val)

    # Add the operation that calculates and applies the gradients to the graph
	train_op = training(loss_val, FLAGS.learning_rate)

    # Add the operations that evaluate the error in predicting the scale on chosen dataset 
 	eval_op = evaluation(scales_est, next_labels)

    # Build the summary Tensor based on the TF collection of Summaries.
   	merged = tf.summary.merge_all()

    # Initialize the global variables
   	init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
   	saver = tf.train.Saver()

   	# Create a session for running operations on the Graph.
   	sess = tf.Session()
   	#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

   	# Instantiate SummaryWriters to output summaries and the Graph.
   	train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
   	validation_writer = tf.summary.FileWriter(FLAGS.log_dir + '/eval')

   	# number of steps per epoch
   	eval_train_steps_per_epoch = int(math.floor(FLAGS.train_size/FLAGS.batch_size))
   	validation_steps_per_epoch = int(math.floor(FLAGS.validation_size/FLAGS.batch_size))

   	# Create the needed handles to feed the handle placeholder
	training_handle = sess.run(training_iterator.string_handle())
	eval_train_handle = sess.run(eval_train_iterator.string_handle())
	validation_handle = sess.run(validation_iterator.string_handle())

    # And then after everything is built:
    # Run the Op to initialize the variables.
   	sess.run(init)

   	# Some variables
   	start_time = time.time()
   	log_frequency = 100
   	log_frequency_comp = log_frequency - 1
   	eval_frequency = 3750
   	eval_frequency_comp = eval_frequency - 1

    # Start the training loop.
  	print('START TRAINING')

   	for steps in xrange(FLAGS.max_steps):	#TODO: change to training epochs
	  	
		# TRAINING
		if steps % log_frequency == log_frequency_comp:
			# Logging for Tensorboard
			run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
	  		run_metadata = tf.RunMetadata()
			summary, _ = sess.run([merged, train_op], feed_dict={handle: training_handle}, options=run_options, run_metadata=run_metadata)
			train_writer.add_run_metadata(run_metadata,'step%d' % steps)
			train_writer.add_summary(summary,steps)
			train_writer.flush()
		else:
			sess.run(train_op, feed_dict={handle: training_handle})

		# Catch end of training "epoch"
		if steps % eval_frequency == eval_frequency_comp:
			duration = time.time() - start_time

			# Print end of "epoch"
			print('Step %d completed. Duration of last %d steps: %d sec' % (steps+1, eval_frequency, duration))


			# EVALUATION RUN ON TRAINING SET
			start_time = time.time()

			# Initialize iterator
			sess.run(eval_train_iterator.initializer)

			# Iterate over training dataset
			eval_loss = 0

			for _ in range(eval_train_steps_per_epoch):
				eval_loss += sess.run(eval_op, feed_dict={handle: eval_train_handle})

			# Calculate average training data evaluation loss
			eval_loss_avg = eval_loss/eval_train_steps_per_epoch

			# Print training data evaluation results
			duration = time.time() - start_time
			print('TRAIN DATA EVALUATION: Num examples: %d  Avg loss: %f Time for evaluation: %d sec' % (FLAGS.train_size, eval_loss_avg, duration))


			# EVALUATION RUN ON TRAINING SET
			start_time = time.time()

			# Initialize iterator
			sess.run(validation_iterator.initializer)

			# Iterate over training dataset
			eval_loss = 0

			for _ in range(validation_steps_per_epoch):
				eval_loss += sess.run(eval_op, feed_dict={handle: validation_handle})

			# Calculate average training data evaluation loss
			eval_loss_avg = eval_loss/validation_steps_per_epoch

			# Print training data evaluation results
			duration = time.time() - start_time
			print('VALIDATION DATA EVALUATION: Num examples: %d  Avg loss: %f Time for evaluation: %d sec' % (FLAGS.validation_size, eval_loss_avg, duration))


			# SAVE A MODEL CHECKPOINT
			checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
			saver.save(sess, checkpoint_file, global_step=steps)


			# Reset time for next training "epoch"
			start_time = time.time()



def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  run_training()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.0001,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--max_steps',
      type=int,
      default=300,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=32,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default='/home/dhri-dz/Documents/HandPose/ScaleNet/Log',
      help='Directory to put the log data.'
  )
  parser.add_argument(
      '--train_size',
      type=int,
      default=120000,
      help='Training data size (number of images).'
  )
  parser.add_argument(
      '--validation_size',
      type=int,
      default=30000,
      help='Validation data size (number of images).'
  )

  FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
