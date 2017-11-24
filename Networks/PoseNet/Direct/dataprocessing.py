# Python 2 Compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports 
import math
import tensorflow as tf
import numpy as np
from utilities import rotate
import cv2


def augmentation_cv(image,joints):

	# Pad image
	image = cv2.copyMakeBorder(image, 64, 64, 64, 64, cv2.BORDER_CONSTANT, value=1.0)

	# Random rotation(counter-clockwise)
	randAngle = -math.pi+2*math.pi*np.random.rand(1)
	rotMat = cv2.getRotationMatrix2D((127.5,127.5), -180*randAngle/math.pi, 1)
	image = cv2.warpAffine(image, rotMat, (256,256),flags=cv2.INTER_NEAREST, borderValue = 1.0)

	# change joint ground truth (rotation)
	joints = joints.reshape(21,3)
	(joints[:,0],joints[:,1]) = rotate((0,0), (joints[:,0],joints[:,1]), -randAngle)
	joints = joints.reshape(63)

	# Scale image and resize to 128x128
	start = int(64)
	end = int(128+1)
	image = cv2.resize(image[start:end, start:end], (128,128), interpolation=cv2.INTER_NEAREST)

	return image, joints


# REQUIRES IMPORT OF rotate_tf from utilities.py
def augmentation(image, joints):

	# Pad image
	image = tf.pad(image,[[64, 64],[64, 64],[0, 0]],"CONSTANT",constant_values=1)

	# Random rotation (counter-clockwise)
	randAngle = math.pi*(tf.random_uniform([1], minval=-180, maxval=180)/180)
	image = tf.contrib.image.rotate(image, -randAngle, interpolation='NEAREST')

	# Change the ground truth (rotation)
	joints = tf.reshape(joints, [21, 3])
	joints_x, joints_y, joints_z = tf.split(joints, num_or_size_splits=3, axis=1)
	joints_x, joints_y, joints_z = rotate_tf((0,0), joints_x, joints_y, joints_z, -randAngle)
	joints = tf.stack([joints_x, joints_y, joints_z], axis=1)
	joints = tf.reshape(joints, [63])
	
	image = tf.image.crop_to_bounding_box(image, 64, 64, 128, 128)
	image = tf.image.resize_images(image, [128, 128], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

	return image, joints


def process_box2scale(image,joints,offset):

    # Extract components
    offset_x, offset_y, offset_z = tf.split(offset, num_or_size_splits=3, axis=0)

    # Threshold the bounding box center offsets (-0.050625m ... 0.050625m --> 0.135m/64*24(margin) for x,y and 100mm = 0.1m for z)
    offset_x = tf.maximum(tf.minimum(offset_x , 0.050625), -0.050625)
    offset_y = tf.maximum(tf.minimum(offset_y , 0.050625), -0.050625)
    offset_z = tf.maximum(tf.minimum(offset_z , 0.1), -0.1)

    # Convert ground truth offset into pixel/normalized depth offset
    offset_x_px = tf.squeeze(offset_x*(64/0.135))
    offset_y_px = tf.squeeze(offset_y*(64/0.135))

    workaround_offset = 0 # to center a certain distance in front of MCP
    offset_z_norm = (offset_z-workaround_offset)/0.235

    # Crop image centered on MCP
    crop_start_x = tf.cast(tf.round(88+offset_x_px-64), tf.int32)
    crop_start_y = tf.cast(tf.round(88+offset_y_px-64), tf.int32)
    image = tf.image.crop_to_bounding_box(image, crop_start_x, crop_start_y, 128, 128)
    image_change = tf.where(tf.less(image, 1.0), (image-offset_z_norm), tf.constant(135/235, dtype=tf.float32, shape=[128, 128, 1]))
    image = tf.where(tf.logical_or(tf.less(image_change, -135/235), tf.greater(image_change, 135/235)), tf.constant(135/235, dtype=tf.float32, shape=[128, 128, 1]), image_change)
    image = image/(135/235)

    # Change the ground truth (--> center on MCP)
    joints = tf.reshape(joints, [21, 3])
    joints_x, joints_y, joints_z = tf.split(joints, num_or_size_splits=3, axis=1)
    joints_x = joints_x - offset_x
    joints_y = joints_y - offset_y
    joints_z = joints_z - offset_z
    joints = tf.stack([joints_x, joints_y, joints_z], axis=1)
    joints = tf.reshape(joints, [63])

    return image, joints


def process_scale2pose(image,handSize):

	# Ensure that hand size is within a desired range (0.5-1.5)
	handSize = tf.maximum(tf.minimum(handSize , 1.5), 0.5)

	# Pad image
	image = tf.pad(image,[[64, 64],[64, 64],[0, 0]],"CONSTANT",constant_values=1)

	# Rescale image to unity hand size
	crop_start = tf.cast(tf.round(128-64*handSize), tf.int32)
	crop_width = tf.cast(tf.round(128*handSize), tf.int32)
	image = tf.image.crop_to_bounding_box(image, crop_start, crop_start, crop_width, crop_width)
	image = tf.image.resize_images(image, [128, 128], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	image_change = tf.where(tf.less(image, 1.0), image/handSize, tf.ones([128, 128, 1]))
	image = tf.where(tf.logical_or(tf.less(image_change, -1), tf.greater(image_change, 1)), tf.ones([128, 128, 1]), image_change)

	return image