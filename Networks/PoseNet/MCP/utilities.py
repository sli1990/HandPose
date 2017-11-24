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

def DH_matrix_mul(theta,d,r,alpha):
    elem11 =  tf.cos(theta)
    elem12 = -tf.sin(theta)*tf.cos(alpha)
    elem13 =  tf.sin(theta)*tf.sin(alpha)
    elem14 =  tf.cos(theta)*r
    elem21 =  tf.sin(theta)
    elem22 =  tf.cos(theta)*tf.cos(alpha)
    elem23 = -tf.cos(theta)*tf.sin(alpha)
    elem24 =  tf.sin(theta)*r
    elem31 =  0.0
    elem32 =  tf.sin(alpha)
    elem33 =  tf.cos(alpha)
    elem34 =  d
    elem41 =  0.0
    elem42 =  0.0
    elem43 =  0.0
    elem44 =  1.0
    line1 = tf.stack([elem11, elem12, elem13, elem14])
    line2 = tf.stack([elem21, elem22, elem23, elem24])
    line3 = tf.stack([elem31, elem32, elem33, elem34])
    line4 = tf.stack([elem41, elem42, elem43, elem44])
    return tf.stack([line1, line2, line3, line4])

def ROT_X(alpha):
    elem11 =  1.0
    elem12 =  0.0
    elem13 =  0.0
    elem14 =  0.0
    elem21 =  0.0
    elem22 =  tf.cos(alpha)
    elem23 = -tf.sin(alpha)
    elem24 =  0.0
    elem31 =  0.0
    elem32 =  tf.sin(alpha)
    elem33 =  tf.cos(alpha)
    elem34 =  0.0
    elem41 =  0.0
    elem42 =  0.0
    elem43 =  0.0
    elem44 =  1.0
    line1 = tf.stack([elem11, elem12, elem13, elem14])
    line2 = tf.stack([elem21, elem22, elem23, elem24])
    line3 = tf.stack([elem31, elem32, elem33, elem34])
    line4 = tf.stack([elem41, elem42, elem43, elem44])
    return tf.stack([line1, line2, line3, line4])

def ROT_Y(gamma):
    elem11 =  tf.cos(gamma)
    elem12 =  0.0
    elem13 =  tf.sin(gamma)
    elem14 =  0.0
    elem21 =  0.0
    elem22 =  1.0
    elem23 =  0.0
    elem24 =  0.0
    elem31 = -tf.sin(gamma)
    elem32 =  0.0
    elem33 =  tf.cos(gamma)
    elem34 =  0.0
    elem41 =  0.0
    elem42 =  0.0
    elem43 =  0.0
    elem44 =  1.0
    line1 = tf.stack([elem11, elem12, elem13, elem14])
    line2 = tf.stack([elem21, elem22, elem23, elem24])
    line3 = tf.stack([elem31, elem32, elem33, elem34])
    line4 = tf.stack([elem41, elem42, elem43, elem44])
    return tf.stack([line1, line2, line3, line4])

def ROT_Z(theta):
    elem11 =  tf.cos(theta)
    elem12 = -tf.sin(theta)
    elem13 =  0.0
    elem14 =  0.0
    elem21 =  tf.sin(theta)
    elem22 =  tf.cos(theta)
    elem23 =  0.0
    elem24 =  0.0
    elem31 =  0.0
    elem32 =  0.0
    elem33 =  1.0
    elem34 =  0.0
    elem41 =  0.0
    elem42 =  0.0
    elem43 =  0.0
    elem44 =  1.0
    line1 = tf.stack([elem11, elem12, elem13, elem14])
    line2 = tf.stack([elem21, elem22, elem23, elem24])
    line3 = tf.stack([elem31, elem32, elem33, elem34])
    line4 = tf.stack([elem41, elem42, elem43, elem44])
    return tf.stack([line1, line2, line3, line4])

def TRANSL(tx,ty,tz):
    elem11 =  1.0
    elem12 =  0.0
    elem13 =  0.0
    elem14 =  tx
    elem21 =  0.0
    elem22 =  1.0
    elem23 =  0.0
    elem24 =  ty
    elem31 =  0.0
    elem32 =  0.0
    elem33 =  1.0
    elem34 =  tz
    elem41 =  0.0
    elem42 =  0.0
    elem43 =  0.0
    elem44 =  1.0
    line1 = tf.stack([elem11, elem12, elem13, elem14])
    line2 = tf.stack([elem21, elem22, elem23, elem24])
    line3 = tf.stack([elem31, elem32, elem33, elem34])
    line4 = tf.stack([elem41, elem42, elem43, elem44])
    return tf.stack([line1, line2, line3, line4])

# phi: x, theta: y, psi: z
def HOM_TRANS(psi,theta,phi,tx,ty,tz):
    elem11 =  tf.cos(theta)*tf.cos(psi)
    elem12 =  tf.cos(theta)*tf.sin(psi)
    elem13 = -tf.sin(theta)
    elem14 =  tx
    elem21 =  tf.sin(phi)*tf.sin(theta)*tf.cos(psi)-tf.cos(phi)*tf.sin(psi)
    elem22 =  tf.sin(phi)*tf.sin(theta)*tf.sin(psi)+tf.cos(phi)*tf.cos(psi)
    elem23 =  tf.sin(phi)*tf.cos(theta)
    elem24 =  ty
    elem31 =  tf.cos(phi)*tf.sin(theta)*tf.cos(psi)+tf.sin(phi)*tf.sin(psi)
    elem32 =  tf.cos(phi)*tf.sin(theta)*tf.sin(psi)-tf.sin(phi)*tf.cos(psi)
    elem33 =  tf.cos(phi)*tf.cos(theta)
    elem34 =  tz
    elem41 =  0.0
    elem42 =  0.0
    elem43 =  0.0
    elem44 =  1.0
    line1 = tf.stack([elem11, elem12, elem13, elem14])
    line2 = tf.stack([elem21, elem22, elem23, elem24])
    line3 = tf.stack([elem31, elem32, elem33, elem34])
    line4 = tf.stack([elem41, elem42, elem43, elem44])
    return tf.stack([line1, line2, line3, line4])