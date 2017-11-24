# Python 2 Compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports 
import math
import tensorflow as tf

def CONV2D(data, filter_size, input_ch, output_ch, padding, relu_bias):
    weights = tf.Variable(tf.random_uniform([filter_size, filter_size, input_ch, output_ch], minval=-math.sqrt(2/(filter_size*filter_size*input_ch)), maxval=math.sqrt(2/(filter_size*filter_size*input_ch))), name='weights')
    if relu_bias == 0:
        biases = tf.Variable(tf.zeros([output_ch]), name='biases')
    else:
        biases = tf.Variable(tf.constant(relu_bias, dtype = tf.float32, shape=[output_ch]), name='biases')
    return tf.nn.relu(tf.nn.conv2d(data, weights, strides=[1, 1, 1, 1], padding=padding) + biases)

def FULLY_CONNECTED(data, inputs, outputs, relu_bias, dropout_var, keep_prob):
    weights = tf.Variable(tf.random_uniform([inputs,outputs], minval=-math.sqrt(2/inputs), maxval=math.sqrt(2/inputs)), name='weights')
    if relu_bias == 0:
        biases = tf.Variable(tf.zeros([outputs]), name='biases')
    else:
        biases = tf.Variable(tf.constant(relu_bias, dtype = tf.float32, shape=[outputs]), name='biases')
    if dropout_var:
        return tf.nn.dropout(tf.nn.relu(tf.matmul(data, weights) + biases) ,keep_prob)
    else:
        return tf.nn.relu(tf.matmul(data, weights) + biases)

def LINEAR_OUTPUT(data, inputs, outputs, bias):
    weights = tf.Variable(tf.random_uniform([inputs,outputs], minval=-math.sqrt(2/inputs), maxval=math.sqrt(2/inputs)), name='weights')
    if bias == 0:
        biases = tf.Variable(tf.zeros([outputs]), name='biases')
    else:
        biases = tf.Variable(tf.constant(bias, dtype = tf.float32, shape=[outputs]), name='biases')
    return tf.matmul(data, weights) + biases

def MAXPOOL2D(data, kernel_size, stride_size, padding):
    return tf.nn.max_pool(data, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride_size, stride_size, 1], padding=padding)

def PADD2D(data, padd_size, padd_value):
    return tf.pad(data,[[0, 0],[padd_size, padd_size],[padd_size, padd_size],[0, 0]],"CONSTANT",constant_values=padd_value)

def RESHAPE2D(data, image_size, channels):
    return tf.reshape(data, [-1, image_size*image_size*channels])