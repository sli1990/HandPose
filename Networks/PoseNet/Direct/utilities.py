# Python 2 Compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports 
import math
import tensorflow as tf

def lrelu(input, alpha):
    return tf.maximum(input, alpha*input)

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

def rotate_tf(origin, jx, jy, jz, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin

    jx_out = ox + tf.cos(angle) * (jx - ox) - tf.sin(angle) * (jy - oy)
    jy_out = oy + tf.sin(angle) * (jx - ox) + tf.cos(angle) * (jy - oy)

    return jx_out, jy_out, jz