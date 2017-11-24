from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import argparse
import os
import sys
import time
import math

import numpy as np
import cv2

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Basic model parameters as external flags.
FLAGS = None

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    
    return qx, qy 

def visualization(image_visu):
    im_flip = tf.transpose(tf.squeeze(image_visu))
    im_flip = tf.reshape(im_flip,[192,192,1])
    return im_flip


def augmentation_cv(image,joint,offset,scale):

    # Pad image
    image = cv2.copyMakeBorder(image, 96, 96, 96, 96, cv2.BORDER_CONSTANT, value=1.0)

    # Random rotation (counter-clockwise)
    randAngle = math.pi*(0/180)
    #randAngle = -math.pi + 2*math.pi*np.random.rand(1)
    rotMat = cv2.getRotationMatrix2D((192,192), -180*randAngle/math.pi, 1)
    image = cv2.warpAffine(image, rotMat, (384,384),flags=cv2.INTER_NEAREST, borderValue = 1.0)

    # Random translate
    randTrans = np.float32([0, 0, 0])/1000
    translMat= np.float32([[1,0,0],[0,1,0]])
    translMat[0,2]= randTrans[1]*(64/0.135)
    translMat[1,2]= randTrans[0]*(64/0.135)
    image = cv2.warpAffine(image, translMat, (384,384),flags=cv2.INTER_NEAREST,borderValue = 1.0)

    # Random scale
    randScale = 2

    joint = joint.reshape(21,3)
    (joint[:,0],joint[:,1]) = rotate((0,0), (joint[:,0],joint[:,1]), -randAngle)
    joint = joint + randTrans
    joint = joint.reshape(63)

    (offset[0],offset[1]) = rotate((0,0), (offset[0],offset[1]), -randAngle)
    offset = offset + randTrans

    msk = np.bitwise_and(image<0.999, image>-0.999)
    image[msk] = np.maximum(np.minimum(image[msk]+randTrans[2]/0.235,1),-1)
    #image[image<1] = np.maximum(np.minimum(image[image<1]+randTrans[2]/0.235,1),-1)

    start = int(round(192-96/randScale))
    end = int(round(192+96/randScale))
    image = cv2.resize(image[start:end, start:end], (192,192), interpolation=cv2.INTER_NEAREST)
    msk = np.bitwise_and(image<0.999, image>-0.999)
    image[msk] =  np.maximum(np.minimum(image[msk]*randScale,1),-1)
    #image[image<1] = np.maximum(np.minimum(image[image<1]*randScale,1),-1)

    # Change the ground truth (scaling)
    joint = joint*randScale
    offset = offset*randScale
    scale = scale*randScale

    return image, joint, offset, scale


def augnet(images_in,joints_in,offsets_in,scales_in,images_aug,joints_aug,offsets_aug,scales_aug):

    # Original image data
    tf.summary.image("image", images_in)
    joints_sq = tf.squeeze(joints_in)
    tf.summary.scalar("x", joints_sq[0])
    tf.summary.scalar("y", joints_sq[1])
    tf.summary.scalar("z", joints_sq[2])
    offsets_sq = tf.squeeze(offsets_in)
    tf.summary.scalar("offx", offsets_sq[0])
    tf.summary.scalar("offy", offsets_sq[1])
    tf.summary.scalar("offz", offsets_sq[2])
    tf.summary.scalar("scale", tf.squeeze(scales_in))

    # Rotate and flip for proper visualization
    images_v = tf.map_fn(visualization, images_in)
    tf.summary.image("images_vis", images_v)

    # Apply the online data augmentation
    images_aug = tf.map_fn(visualization, images_aug)   # for visualization
    tf.summary.image("images_aug", images_aug)
    joints_aug_sq = tf.squeeze(joints_aug)
    tf.summary.scalar("x_aug", joints_aug_sq[0])
    tf.summary.scalar("y_aug", joints_aug_sq[1])
    tf.summary.scalar("z_aug", joints_aug_sq[2])
    offsets_aug_sq = tf.squeeze(offsets_aug)
    tf.summary.scalar("offx_aug", offsets_aug_sq[0])
    tf.summary.scalar("offy_aug", offsets_aug_sq[1])
    tf.summary.scalar("offz_aug", offsets_aug_sq[2])
    tf.summary.scalar("scale_aug", tf.squeeze(scales_aug))

    return images_in, joints_in, offsets_in, scales_in, images_aug, joints_aug, offsets_aug, scales_aug


def evaluation(input1,input2,input3,input4,input5,input6,input7,input8):
    return input1,input2,input3,input4,input5,input6,input7,input8


def parse_function_augment(example_proto):

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

    images = image_reshaped
    joints = tf.cast(joint_dec, tf.float32)
    offsets = tf.cast(offset_dec, tf.float32)
    scales = tf.cast(handScale_dec, tf.float32)

    images_aug, joints_aug, offsets_aug, scales_aug = tf.py_func(augmentation_cv,[images,joints,offsets,scales],[tf.float32, tf.float32, tf.float32, tf.float32])

    return images_aug, joints_aug, offsets_aug, scales_aug


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

    images = image_reshaped
    joints = tf.cast(joint_dec, tf.float32)
    offsets = tf.cast(offset_dec, tf.float32)
    scales = tf.cast(handScale_dec, tf.float32)

    return images, joints, offsets, scales


def run_augmentation():

    # Data location
    filenames = ["C:/Users\Jan\Desktop\Aug_CV\AugTest"]

    # Define the training with Dataset API
    dataset = tf.contrib.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_function_augment)
    dataset = dataset.batch(FLAGS.batch_size)

    dataset2 = tf.contrib.data.TFRecordDataset(filenames)
    dataset2 = dataset2.map(parse_function)
    dataset2 = dataset2.batch(FLAGS.batch_size) 
    
    # Create iterator
    iterator = dataset.make_one_shot_iterator()
    iterator2 = dataset2.make_one_shot_iterator()
    next_images_aug, next_joints_aug, next_offsets_aug, next_scales_aug = iterator.get_next()
    next_images, next_joints, next_offsets, next_scales = iterator2.get_next()

    # Build Graph
    im_out, joint_out, offset_out, scale_out, im_out_aug, joint_out_aug, offset_out_aug, scale_out_aug = augnet(next_images, next_joints, next_offsets, next_scales, next_images_aug, next_joints_aug, next_offsets_aug, next_scales_aug)

    # Add the operations that evaluate
    eval_op = evaluation(im_out, joint_out, offset_out, scale_out, im_out_aug, joint_out_aug, offset_out_aug, scale_out_aug)

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
      default='C:/Users\Jan\Desktop\Aug_CV\Log',
      help='Directory to put the log data.'
  )

  FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)