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
IMAGE_SIZE = 192
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

# Basic model parameters as external flags.
FLAGS = None

def rotate(origin, jx, jy, jz, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin

    jx_out = ox + tf.cos(angle) * (jx - ox) - tf.sin(angle) * (jy - oy)
    jy_out = oy + tf.sin(angle) * (jx - ox) + tf.cos(angle) * (jy - oy)

    return jx_out, jy_out, jz#tf.stack([jx_out, jy_out, jz], axis=1)

def visualization(image):

	im_rot = tf.image.rot90(image, k=3)
	im_flip = tf.image.flip_left_right(im_rot)
	
	return im_flip


def augmentation(input): # add possibility for leaving the image unchanged

	# Extract components
	image = input[0]
	joints = input[1]

	# Pad image
	image = tf.pad(image,[[96, 96],[96, 96],[0, 0]],"CONSTANT",constant_values=1)

	# Random rotation (counter-clockwise)
	#randAngle = math.pi*(0/180)
	randAngle = math.pi*(tf.random_uniform([1], minval=-180, maxval=180)/180)
	image = tf.contrib.image.rotate(image, randAngle, interpolation='NEAREST')

	# Random translate
	#randTrans = tf.constant([20.0, 0.0])
	#randTransZ = 0.0
	randTrans = tf.maximum(tf.minimum(tf.cast(tf.round(tf.random_normal([2], mean=0.0, stddev=5.0)),tf.float32), 15.0), -15.0)
	randTransZ = -0.025 + 0.05*tf.squeeze(tf.maximum(tf.minimum(tf.random_normal([1], mean=0.5, stddev=0.1), 1.0), 0.0))

	# Random scale
	#randScale = 1.0
	randScale = tf.maximum(tf.minimum(tf.random_normal([1], mean=1.0, stddev=0.1), 1.25), 0.75)
	crop_start_x = tf.cast(tf.round(192-randTrans[0]-96/tf.squeeze(randScale)), tf.int32)
	crop_start_y = tf.cast(tf.round(192-randTrans[1]-96/tf.squeeze(randScale)), tf.int32)
	crop_width = tf.cast(tf.round(192/tf.squeeze(randScale)), tf.int32)
	image = tf.image.crop_to_bounding_box(image, crop_start_x, crop_start_y, crop_width, crop_width)
	image = tf.image.resize_images(image, [192, 192], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	image_change = tf.where(tf.less(image, 1.0), (image+(randTransZ/0.25))*randScale, tf.ones([192, 192, 1]))
	image = tf.where(tf.logical_or(tf.less(image_change, -1), tf.greater(image_change, 1)), tf.ones([192, 192, 1]), image_change)

	# Change the ground truth (rotation)
	joints = tf.reshape(joints, [21, 3])
	joints_x, joints_y, joints_z = tf.split(joints, num_or_size_splits=3, axis=1)
	joints_x, joints_y, joints_z = rotate((0,0), joints_x, joints_y, joints_z, randAngle)

	# Change the ground truth (translation: x and y flipped due to array indexing)
	joints_x = joints_x + randTrans[1]*(0.135/64)
	joints_y = joints_y + randTrans[0]*(0.135/64)
	joints_z = joints_z + randTransZ
	joints = tf.stack([joints_x, joints_y, joints_z], axis=1)
	joints = tf.reshape(joints, [63])

	# Change the ground truth (scaling)
	joints = joints*randScale

	return image, joints#, randTrans[0], randTrans[1]


def process_box2scale(input):

	# Extract components
	image = input[0]
	joints = input[1]
	offset = input[2]
	offset_x, offset_y, offset_z = tf.split(offset, num_or_size_splits=3, axis=0)

	# Ensure that offset is within a desired range (-0.0675 ... 0.0675 --> 0.135/64*32(margin) for x,y and 100mm = 0.1m for z)
	offset_x = tf.maximum(tf.minimum(offset_x , 0.0675), -0.0675)
	offset_y = tf.maximum(tf.minimum(offset_y , 0.0675), -0.0675)
	offset_z = tf.maximum(tf.minimum(offset_z , 0.1), -0.1)

	# Convert ground truth offset into pixel/normalized depth offset
	offset_x_px = tf.squeeze(offset_x*(64/0.135))
	offset_y_px = tf.squeeze(offset_y*(64/0.135))

	workaround_offset = 0 # to center a certain distance in front of MCP
	offset_z_norm = offset_z/0.235 - workaround_offset

	# Crop image centered on MCP
	image_c = tf.image.crop_to_bounding_box(image, 32, 32, 128, 128)
	crop_start_x = tf.cast(tf.round(96+offset_x_px-64), tf.int32)
	crop_start_y = tf.cast(tf.round(96+offset_y_px-64), tf.int32)
	image = tf.image.crop_to_bounding_box(image, crop_start_x, crop_start_y, 128, 128)	# x and y flipped due to tensor indexing
	image_change = tf.where(tf.less(image, 1.0), (image-offset_z_norm), tf.constant(135/235, dtype=tf.float32, shape=[128, 128, 1]))
	image = tf.where(tf.logical_or(tf.less(image_change, -135/235-offset_z_norm), tf.greater(image_change, 135/235-offset_z_norm)), tf.constant(135/235, dtype=tf.float32, shape=[128, 128, 1]), image_change)
	image = image/(135/235)

	# Change the ground truth (--> center on MCP)
	joints = tf.reshape(joints, [21, 3])
	joints_x, joints_y, joints_z = tf.split(joints, num_or_size_splits=3, axis=1)
	joints_x = joints_x - offset_x
	joints_y = joints_y - offset_y
	joints_z = joints_z - offset_z
	joints = tf.stack([joints_x, joints_y, joints_z], axis=1)
	joints = tf.reshape(joints, [63])

	# --> must center3ds be changed?

	return image, image_c, joints, offset


def process_scale2pose(input):

	# Extract components
	image = input[0]
	joints = input[1]
	handSize = input[2]

	# Ensure that hand size is within a desired range (0.5-1.5)
	handSize = tf.maximum(tf.minimum(tf.squeeze(handSize) , 1.5), 0.5)

	# Pad image
	image = tf.pad(image,[[64, 64],[64, 64],[0, 0]],"CONSTANT",constant_values=1)

	# Rescale image to unity hand size
	crop_start = tf.cast(tf.round(128-64*handSize), tf.int32)
	crop_width = tf.cast(tf.round(128*handSize), tf.int32)
	image = tf.image.crop_to_bounding_box(image, crop_start, crop_start, crop_width, crop_width)
	image = tf.image.resize_images(image, [128, 128], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	image_change = tf.where(tf.less(image, 1.0), image/handSize, tf.ones([128, 128, 1]))
	image = tf.where(tf.logical_or(tf.less(image_change, -1), tf.greater(image_change, 1)), tf.ones([128, 128, 1]), image_change)

	# Change the ground truth (scaling)
	#--> clarify if needed

	return image, handSize


def augnet(images, joints):

	# Original image data
	tf.summary.image("image", images)
	joints_sq = tf.squeeze(joints)
	tf.summary.scalar("x", joints_sq[0])
	tf.summary.scalar("y", joints_sq[1])
	tf.summary.scalar("z", joints_sq[2])

	# Rotate and flip for proper visualization
	images = tf.map_fn(visualization, images)
	tf.summary.image("images_vis", images)

	# Apply the online data augmentation
	#images_aug, transx, transy = tf.map_fn(augmentation, images, dtype=(tf.float32, tf.int32, tf.int32))
	images_aug, joints_aug = tf.map_fn(augmentation, [images, joints], dtype=(tf.float32, tf.float32))
	tf.summary.image("images_aug", images_aug)
	#tf.summary.tensor_summary("shift", trans)
	#tf.summary.scalar("shiftx", tf.squeeze(transx))
	#tf.summary.scalar("shifty", tf.squeeze(transy))
	joints_aug_sq = tf.squeeze(joints_aug)
	tf.summary.scalar("x_aug", joints_aug_sq[0])
	tf.summary.scalar("y_aug", joints_aug_sq[1])
	tf.summary.scalar("z_aug", joints_aug_sq[2])

  	return images_aug, joints_aug


def b2sNet(images, joints, offsets):

	# Original image data
	tf.summary.image("image", images)
	joints_sq = tf.squeeze(joints)
	tf.summary.scalar("x", joints_sq[0])
	tf.summary.scalar("y", joints_sq[1])
	tf.summary.scalar("z", joints_sq[2])

	# Apply the data processing
	images_p, images_c, joints_p, offsets_p = tf.map_fn(process_box2scale, [images, joints, offsets], dtype=(tf.float32, tf.float32, tf.float32, tf.float32))
	# Rotate and flip for proper visualization
	images_p = tf.map_fn(visualization, images_p)
	images_c = tf.map_fn(visualization, images_c)
	
	tf.summary.image("images_p", images_p)
	tf.summary.scalar("zero", tf.squeeze(images_p[0,0,1]))
	joints_p_sq = tf.squeeze(joints_p)
	tf.summary.scalar("x_p", joints_p_sq[0])
	tf.summary.scalar("y_p", joints_p_sq[1])
	tf.summary.scalar("z_p", joints_p_sq[2])
	offsets_p_sq = tf.squeeze(offsets_p)
	tf.summary.scalar("x_off_p", offsets_p_sq[0])
	tf.summary.scalar("y_off_p", offsets_p_sq[1])
	tf.summary.scalar("z_off_p", offsets_p_sq[2])

	tf.summary.image("images_vis", images_c)
	tf.summary.scalar("zero_orig", tf.squeeze(images_c[0,0,1]))

  	return images_p, joints_p, offsets_p


def s2pNet(images, joints, scales):

	# Original image data
	tf.summary.image("image", images)
	joints_sq = tf.squeeze(joints)

	# Visualization
	images_c = tf.map_fn(visualization, images)
	tf.summary.image("images_vis", images_c)

	# Apply the data processing
	images_p, scales_p = tf.map_fn(process_scale2pose, [images, joints, scales], dtype=(tf.float32, tf.float32))
	
	# Rotate and flip for proper visualization
	images_p = tf.map_fn(visualization, images_p)
	tf.summary.image("images_p", images_p)
	tf.summary.scalar("scale", tf.squeeze(scales_p))

  	return images_p, joints, scales_p


def evaluation(input):
	return input


def parse_function(example_proto):

	# Parse through features and extract byte string
	parsed_features = tf.parse_single_example(example_proto,features ={
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
	image_reshaped = tf.image.crop_to_bounding_box(image_reshaped, 32, 32, 128, 128)

	return image_reshaped, tf.cast(joint_dec, tf.float32), tf.cast(handScale_dec, tf.float32)


def run_augmentation():

	# Data location
	filenames = ["/home/dhri-dz/Documents/HandPose/Augmentation/AugTest"]
	#filenames = ["/home/dhri-dz/Hands2017Train_1"]

  	# Define the training with Dataset API
	dataset = tf.contrib.data.TFRecordDataset(filenames)
	dataset = dataset.map(parse_function)
	dataset = dataset.batch(FLAGS.batch_size)	
	
	# Create iterator
	iterator = dataset.make_one_shot_iterator()
	next_images, next_joints, next_scales = iterator.get_next()

	# Build Graph
	im_out, joint_out, off_out = s2pNet(next_images, next_joints, next_scales)

    # Add the operations that evaluate
 	eval_op = evaluation(im_out)

    # Build the summary Tensor based on the TF collection of Summaries.
   	merged = tf.summary.merge_all()

    # Initialize the global variables
   	init = tf.global_variables_initializer()

   	# Create a session for running operations on the Graph.
   	sess = tf.Session()

   	# Instantiate SummaryWriters to output summaries and the Graph.
   	writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    # And then after everything is built:
    # Run the Op to initialize the variables.
   	sess.run(init)

    # Start
  	print('START')

   	for steps in range(FLAGS.max_steps):
	  	
		# Logging for Tensorboard
		summary, image_out = sess.run([merged, eval_op])
		writer.add_summary(summary,steps)
		writer.flush()


def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  run_augmentation()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--max_steps',
      type=int,
      default=12,
      help='Number of steps to run.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=1,
      help='Batch size.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default='/home/dhri-dz/Documents/HandPose/Augmentation/Log',
      help='Directory to put the log data.'
  )

  FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)