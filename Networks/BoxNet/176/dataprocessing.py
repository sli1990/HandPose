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


def augmentation_cv(image,joints,offsets,scales):

	# Pad image
	image = cv2.copyMakeBorder(image, 88, 88, 88, 88, cv2.BORDER_CONSTANT, value=1.0)

	# Random rotation(counter-clockwise)
	randAngle = -math.pi+2*math.pi*np.random.rand(1)
	rotMat = cv2.getRotationMatrix2D((175.5,175.5), -180*randAngle/math.pi, 1)
	image = cv2.warpAffine(image, rotMat, (352,352),flags=cv2.INTER_NEAREST, borderValue = 1.0)

	# change joint ground truth (rotation)
	joints = joints.reshape(21,3)
	(joints[:,0],joints[:,1]) = rotate((0,0), (joints[:,0],joints[:,1]), -randAngle)

	# change offset ground truth (rotation)
	(offsets[0],offsets[1]) = rotate((0,0), (offsets[0],offsets[1]), -randAngle)

	# Random translate
	randTrans = np.float32(np.maximum(np.minimum(np.random.normal(0.0, 4.0, (3,)),15.0),-15.0)/1000)

	# x,y translation (flipped due to OpenCV convention)
	translMat= np.float32([[1,0,0],[0,1,0]])
	translMat[0,2] = randTrans[1]*(64/0.135)
	translMat[1,2] = randTrans[0]*(64/0.135)
	image = cv2.warpAffine(image, translMat, (352,352),flags=cv2.INTER_NEAREST,borderValue = 1.0)

	# z translation
	msk = np.bitwise_and(image<0.999, image>-0.999)
	image[msk] = np.maximum(np.minimum(image[msk]+randTrans[2]/0.235,1),-1)

	# change joint ground truth (translation)
	joints = joints + randTrans
	joints = joints.reshape(63)

	# change offset ground truth (translation)
	offsets = offsets + randTrans

	# Random scale
	randScale = np.maximum(np.minimum(np.random.normal(1.0, 0.075),1.25),0.75)

	# Scale image and resize to 128x128
	start = int(round(176-88/randScale))
	end = int(round(176+1+88/randScale))
	image = cv2.resize(image[start:end, start:end], (176,176), interpolation=cv2.INTER_NEAREST)
	msk = np.bitwise_and(image<0.999, image>-0.999)
	image[msk] = np.maximum(np.minimum(image[msk]*randScale,1),-1)

	# change ground truth (scaling)
	joints = joints*randScale
	offsets = offsets*randScale
	scales = scales*randScale

	return image, joints, offsets, scales

# REQUIRES IMPORT OF rotate_tf from utilities.py
def augmentation(image, joints, offsets, scales):

	# Pad image
	image = tf.pad(image,[[88, 88],[88, 88],[0, 0]],"CONSTANT",constant_values=1)

	# Random rotation (counter-clockwise)
	randAngle = math.pi*(tf.random_uniform([1], minval=-180, maxval=180)/180)
	image = tf.contrib.image.rotate(image, -randAngle, interpolation='NEAREST')

	# Random translate
	randTrans = tf.maximum(tf.minimum(tf.cast(tf.round(tf.random_normal([2], mean=0.0, stddev=4.0)),tf.float32), 15.0), -15.0)/1000
	randTransZ = tf.squeeze(tf.maximum(tf.minimum(tf.random_normal([1], mean=0.0, stddev=4.0), 15.0), -15.0))/1000

	# Change the ground truth (rotation)
	joints = tf.reshape(joints, [21, 3])
	joints_x, joints_y, joints_z = tf.split(joints, num_or_size_splits=3, axis=1)
	joints_x, joints_y, joints_z = rotate_tf((0,0), joints_x, joints_y, joints_z, -randAngle)

	offsets_x, offsets_y, offsets_z = tf.split(offsets, num_or_size_splits=3, axis=0)
	offsets_x, offsets_y, offsets_z = rotate_tf((0,0), offsets_x, offsets_y, offsets_z, -randAngle)

	# Change the ground truth (translation)
	joints_x = joints_x + randTrans[0]
	joints_y = joints_y + randTrans[1]
	joints_z = joints_z + randTransZ
	joints = tf.stack([joints_x, joints_y, joints_z], axis=1)
	joints = tf.reshape(joints, [63])

	offsets_x = offsets_x + randTrans[0]
	offsets_y = offsets_y + randTrans[1]
	offsets_z = offsets_z + randTransZ
	offsets = tf.squeeze(tf.stack([offsets_x, offsets_y, offsets_z], axis=1))

	randTrans = randTrans*(64/0.135)

	# Random scale
	randScale = tf.squeeze(tf.maximum(tf.minimum(tf.random_normal([1], mean=1.0, stddev=0.075), 1.25), 0.75))
	
	crop_start_x = tf.cast(tf.round(176-randTrans[0]-88/randScale), tf.int32)
	crop_start_y = tf.cast(tf.round(176-randTrans[1]-88/randScale), tf.int32)
	crop_width = tf.cast(tf.round(176/randScale), tf.int32)
	image = tf.image.crop_to_bounding_box(image, crop_start_x, crop_start_y, crop_width, crop_width)
	image = tf.image.resize_images(image, [176, 176], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	image_change = tf.where(tf.less(image, 1.0), (image+(randTransZ/0.235))*randScale, tf.ones([176, 176, 1]))
	image = tf.where(tf.logical_or(tf.less(image_change, -1), tf.greater(image_change, 1)), tf.ones([176, 176, 1]), image_change)

	# Change the ground truth (scaling)
	joints = joints*randScale
	offsets = offsets*randScale
	scales = scales*randScale

	return image, joints, offsets, scales