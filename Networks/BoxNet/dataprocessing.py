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
	image = cv2.copyMakeBorder(image, 64, 64, 64, 64, cv2.BORDER_CONSTANT, value=1.0)

	# Random rotation(counter-clockwise)
	randAngle = -math.pi+2*math.pi*np.random.rand(1)
	rotMat = cv2.getRotationMatrix2D((127.5,127.5), -180*randAngle/math.pi, 1)
	image = cv2.warpAffine(image, rotMat, (256,256),flags=cv2.INTER_NEAREST, borderValue = 1.0)

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
	image = cv2.warpAffine(image, translMat, (256,256),flags=cv2.INTER_NEAREST,borderValue = 1.0)

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
	start = int(round(128-64/randScale))
	end = int(round(128+64/randScale))
	image = cv2.resize(image[start:end, start:end], (128,128), interpolation=cv2.INTER_NEAREST)
	msk = np.bitwise_and(image<0.999, image>-0.999)
	image[msk] = np.maximum(np.minimum(image[msk]*randScale,1),-1)

	# change ground truth (scaling)
	joints = joints*randScale
	offsets = offsets*randScale
	scales = scales*randScale

	return image, joints, offsets, scales