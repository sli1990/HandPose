# Python 2 Compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports 
import math
import tensorflow as tf
from layers import *

def ScaleLayer(input):

    joints = input[0]
    scale = input[1]

    return joints*scale


def PoseNet(images,joints,scales):
	"""PoseNet model
  	Args:	
  		images: image batch to be processed
  		joints: joint ground truth 
  		scales: hand scale ground truth
  	Returns:
    	pose3D: 3D pose of all 21 joints (63 elements)
  	"""

  	# Placeholders
  	#keep_prob = tf.placeholder(tf.float32)

  	# Scope vairables for PoseNet
  	with tf.name_scope('posenet'):

	  	# Padding Layer (with value 1) to maintain size
	  	with tf.name_scope('padd'):
	  		padded_input = PADD2D(images, 2, 1)

	  	# Convolutional Layer 1
	  	with tf.name_scope('convolution1'):
	  		hidden1 = CONV2D(padded_input, 5, 1, 12, 'VALID', 0)

	  	# Pooling Layer 1
	  	with tf.name_scope('pool1'):
	  		hidden1_pool = MAXPOOL2D(hidden1, 4, 4, 'SAME')

	  	# Padding Layer (with value 1) to maintain size
	  	with tf.name_scope('padd2'):
	  		padded_2 = PADD2D(hidden1_pool, 2, 1)

	  	# Convolutional Layer 2
	  	with tf.name_scope('convolution2'):
	  		hidden2 = CONV2D(padded_2, 5, 12, 12, 'VALID', 0)

	  	# Pooling Layer 2
	  	with tf.name_scope('pool2'):
	  		hidden2_pool = MAXPOOL2D(hidden2, 2, 2, 'SAME')

	  	# Padding Layer (with value 1) to maintain size
	  	with tf.name_scope('padd3'):
	  		padded_3 = PADD2D(hidden2_pool, 1, 1)

	  	# Convolutional Layer 3
	  	with tf.name_scope('convolution3'):
	  		hidden3 = CONV2D(padded_3, 3, 12, 12, 'VALID', 0)

	  	# Flatten output from convolutional layers
	  	with tf.name_scope('flatten'):
	  		hidden3_flat = RESHAPE2D(hidden3, 16, 12)

	  	# Fully Connected Layer 1
	  	with tf.name_scope('fc1'):
	  		hidden4_a = FULLY_CONNECTED(hidden3_flat, 16*16*12, 1024, 0, False, 0.0)

	  	# Fully Connected Layer 2
	  	with tf.name_scope('fc2'):
	  		hidden5_a = FULLY_CONNECTED(hidden4_a, 1024, 1024, 0, False, 0.0)

	  	# Linear Output Layer
	  	with tf.name_scope('linear'):
	  		pose3D = LINEAR_OUTPUT(hidden5_a, 1024, 63, 0)

		# Scale up joint 3D pose values
		with tf.name_scope('scaleUp'):
			pose3D = tf.map_fn(ScaleLayer, [pose3D, scales], dtype=tf.float32, parallel_iterations=32)
  	
  	return pose3D


def loss_posenet(pose3D, poseGT):
	"""Calculates the combined loss of the PoseNet.
	Args:
		pose3D: 3D pose estimated by PoseNet
		poseGT: ground truth of 3D pose
	Returns:
		loss: Loss tensor of type float.
	"""

	# 3D pose loss
	poseLoss = tf.nn.l2_loss(pose3D-poseGT)

	# Compositional pose loss
	pose3D_reshaped = tf.reshape(pose3D, [tf.shape(pose3D)[0] ,3, 21])
	poseGT_reshaped = tf.reshape(poseGT, [tf.shape(poseGT)[0] ,3, 21])

	jointVec1  = pose3D_reshaped[:,:,0]
	jointVec2  = pose3D_reshaped[:,:,1]
	jointVec3  = pose3D_reshaped[:,:,2]
	jointVec4  = pose3D_reshaped[:,:,3]
	jointVec5  = pose3D_reshaped[:,:,4]
	jointVec6  = pose3D_reshaped[:,:,5]
	jointVec7  = pose3D_reshaped[:,:,6]
	jointVec8  = pose3D_reshaped[:,:,7]
	jointVec9  = pose3D_reshaped[:,:,8]
	jointVec10 = pose3D_reshaped[:,:,9]
	jointVec11 = pose3D_reshaped[:,:,10]
	jointVec12 = pose3D_reshaped[:,:,11]
	jointVec13 = pose3D_reshaped[:,:,12]
	jointVec14 = pose3D_reshaped[:,:,13]
	jointVec15 = pose3D_reshaped[:,:,14]
	jointVec16 = pose3D_reshaped[:,:,15]
	jointVec17 = pose3D_reshaped[:,:,16]
	jointVec18 = pose3D_reshaped[:,:,17]
	jointVec19 = pose3D_reshaped[:,:,18]
	jointVec20 = pose3D_reshaped[:,:,19]
	jointVec21 = pose3D_reshaped[:,:,20]

	jointVec1_GT  = poseGT_reshaped[:,:,0]
	jointVec2_GT  = poseGT_reshaped[:,:,1]
	jointVec3_GT  = poseGT_reshaped[:,:,2]
	jointVec4_GT  = poseGT_reshaped[:,:,3]
	jointVec5_GT  = poseGT_reshaped[:,:,4]
	jointVec6_GT  = poseGT_reshaped[:,:,5]
	jointVec7_GT  = poseGT_reshaped[:,:,6]
	jointVec8_GT  = poseGT_reshaped[:,:,7]
	jointVec9_GT  = poseGT_reshaped[:,:,8]
	jointVec10_GT = poseGT_reshaped[:,:,9]
	jointVec11_GT = poseGT_reshaped[:,:,10]
	jointVec12_GT = poseGT_reshaped[:,:,11]
	jointVec13_GT = poseGT_reshaped[:,:,12]
	jointVec14_GT = poseGT_reshaped[:,:,13]
	jointVec15_GT = poseGT_reshaped[:,:,14]
	jointVec16_GT = poseGT_reshaped[:,:,15]
	jointVec17_GT = poseGT_reshaped[:,:,16]
	jointVec18_GT = poseGT_reshaped[:,:,17]
	jointVec19_GT = poseGT_reshaped[:,:,18]
	jointVec20_GT = poseGT_reshaped[:,:,19]
	jointVec21_GT = poseGT_reshaped[:,:,20]

	compLoss1   = tf.nn.l2_loss((jointVec1 -jointVec2) -(jointVec1_GT -jointVec2_GT))
	compLoss2   = tf.nn.l2_loss((jointVec1 -jointVec3) -(jointVec1_GT -jointVec3_GT))
	compLoss3   = tf.nn.l2_loss((jointVec1 -jointVec4) -(jointVec1_GT -jointVec4_GT))
	compLoss4   = tf.nn.l2_loss((jointVec1 -jointVec5) -(jointVec1_GT -jointVec5_GT))
	compLoss5   = tf.nn.l2_loss((jointVec1 -jointVec6) -(jointVec1_GT -jointVec6_GT))
	compLoss6   = tf.nn.l2_loss((jointVec1 -jointVec7) -(jointVec1_GT -jointVec7_GT))
	compLoss7   = tf.nn.l2_loss((jointVec1 -jointVec8) -(jointVec1_GT -jointVec8_GT))
	compLoss8   = tf.nn.l2_loss((jointVec1 -jointVec9) -(jointVec1_GT -jointVec9_GT))
	compLoss9   = tf.nn.l2_loss((jointVec1 -jointVec10)-(jointVec1_GT -jointVec10_GT))
	compLoss10  = tf.nn.l2_loss((jointVec1 -jointVec11)-(jointVec1_GT -jointVec11_GT))
	compLoss11  = tf.nn.l2_loss((jointVec1 -jointVec12)-(jointVec1_GT -jointVec12_GT))
	compLoss12  = tf.nn.l2_loss((jointVec1 -jointVec13)-(jointVec1_GT -jointVec13_GT))
	compLoss13  = tf.nn.l2_loss((jointVec1 -jointVec14)-(jointVec1_GT -jointVec14_GT))
	compLoss14  = tf.nn.l2_loss((jointVec1 -jointVec15)-(jointVec1_GT -jointVec15_GT))
	compLoss15  = tf.nn.l2_loss((jointVec1 -jointVec16)-(jointVec1_GT -jointVec16_GT))
	compLoss16  = tf.nn.l2_loss((jointVec1 -jointVec17)-(jointVec1_GT -jointVec17_GT))
	compLoss17  = tf.nn.l2_loss((jointVec1 -jointVec18)-(jointVec1_GT -jointVec18_GT))
	compLoss18  = tf.nn.l2_loss((jointVec1 -jointVec19)-(jointVec1_GT -jointVec19_GT))
	compLoss19  = tf.nn.l2_loss((jointVec1 -jointVec20)-(jointVec1_GT -jointVec20_GT))
	compLoss20  = tf.nn.l2_loss((jointVec1 -jointVec21)-(jointVec1_GT -jointVec21_GT))

	compLoss21  = tf.nn.l2_loss((jointVec2 -jointVec3) -(jointVec2_GT -jointVec3_GT))
	compLoss22  = tf.nn.l2_loss((jointVec2 -jointVec4) -(jointVec2_GT -jointVec4_GT))
	compLoss23  = tf.nn.l2_loss((jointVec2 -jointVec5) -(jointVec2_GT -jointVec5_GT))
	compLoss24  = tf.nn.l2_loss((jointVec2 -jointVec6) -(jointVec2_GT -jointVec6_GT))
	compLoss25  = tf.nn.l2_loss((jointVec2 -jointVec7) -(jointVec2_GT -jointVec7_GT))
	compLoss26  = tf.nn.l2_loss((jointVec2 -jointVec8) -(jointVec2_GT -jointVec8_GT))
	compLoss27  = tf.nn.l2_loss((jointVec2 -jointVec9) -(jointVec2_GT -jointVec9_GT))
	compLoss28  = tf.nn.l2_loss((jointVec2 -jointVec10)-(jointVec2_GT -jointVec10_GT))
	compLoss29  = tf.nn.l2_loss((jointVec2 -jointVec11)-(jointVec2_GT -jointVec11_GT))
	compLoss30  = tf.nn.l2_loss((jointVec2 -jointVec12)-(jointVec2_GT -jointVec12_GT))
	compLoss31  = tf.nn.l2_loss((jointVec2 -jointVec13)-(jointVec2_GT -jointVec13_GT))
	compLoss32  = tf.nn.l2_loss((jointVec2 -jointVec14)-(jointVec2_GT -jointVec14_GT))
	compLoss33  = tf.nn.l2_loss((jointVec2 -jointVec15)-(jointVec2_GT -jointVec15_GT))
	compLoss34  = tf.nn.l2_loss((jointVec2 -jointVec16)-(jointVec2_GT -jointVec16_GT))
	compLoss35  = tf.nn.l2_loss((jointVec2 -jointVec17)-(jointVec2_GT -jointVec17_GT))
	compLoss36  = tf.nn.l2_loss((jointVec2 -jointVec18)-(jointVec2_GT -jointVec18_GT))
	compLoss37  = tf.nn.l2_loss((jointVec2 -jointVec19)-(jointVec2_GT -jointVec19_GT))
	compLoss38  = tf.nn.l2_loss((jointVec2 -jointVec20)-(jointVec2_GT -jointVec20_GT))
	compLoss39  = tf.nn.l2_loss((jointVec2 -jointVec21)-(jointVec2_GT -jointVec21_GT))

	compLoss40  = tf.nn.l2_loss((jointVec3 -jointVec4) -(jointVec3_GT -jointVec4_GT))
	compLoss41  = tf.nn.l2_loss((jointVec3 -jointVec5) -(jointVec3_GT -jointVec5_GT))
	compLoss42  = tf.nn.l2_loss((jointVec3 -jointVec6) -(jointVec3_GT -jointVec6_GT))
	compLoss43  = tf.nn.l2_loss((jointVec3 -jointVec7) -(jointVec3_GT -jointVec7_GT))
	compLoss44  = tf.nn.l2_loss((jointVec3 -jointVec8) -(jointVec3_GT -jointVec8_GT))
	compLoss45  = tf.nn.l2_loss((jointVec3 -jointVec9) -(jointVec3_GT -jointVec9_GT))
	compLoss46  = tf.nn.l2_loss((jointVec3 -jointVec10)-(jointVec3_GT -jointVec10_GT))
	compLoss47  = tf.nn.l2_loss((jointVec3 -jointVec11)-(jointVec3_GT -jointVec11_GT))
	compLoss48  = tf.nn.l2_loss((jointVec3 -jointVec12)-(jointVec3_GT -jointVec12_GT))
	compLoss49  = tf.nn.l2_loss((jointVec3 -jointVec13)-(jointVec3_GT -jointVec13_GT))
	compLoss50  = tf.nn.l2_loss((jointVec3 -jointVec14)-(jointVec3_GT -jointVec14_GT))
	compLoss51  = tf.nn.l2_loss((jointVec3 -jointVec15)-(jointVec3_GT -jointVec15_GT))
	compLoss52  = tf.nn.l2_loss((jointVec3 -jointVec16)-(jointVec3_GT -jointVec16_GT))
	compLoss53  = tf.nn.l2_loss((jointVec3 -jointVec17)-(jointVec3_GT -jointVec17_GT))
	compLoss54  = tf.nn.l2_loss((jointVec3 -jointVec18)-(jointVec3_GT -jointVec18_GT))
	compLoss55  = tf.nn.l2_loss((jointVec3 -jointVec19)-(jointVec3_GT -jointVec19_GT))
	compLoss56  = tf.nn.l2_loss((jointVec3 -jointVec20)-(jointVec3_GT -jointVec20_GT))
	compLoss57  = tf.nn.l2_loss((jointVec3 -jointVec21)-(jointVec3_GT -jointVec21_GT))

	compLoss58  = tf.nn.l2_loss((jointVec4 -jointVec5) -(jointVec4_GT -jointVec5_GT))
	compLoss59  = tf.nn.l2_loss((jointVec4 -jointVec6) -(jointVec4_GT -jointVec6_GT))
	compLoss60  = tf.nn.l2_loss((jointVec4 -jointVec7) -(jointVec4_GT -jointVec7_GT))
	compLoss61  = tf.nn.l2_loss((jointVec4 -jointVec8) -(jointVec4_GT -jointVec8_GT))
	compLoss62  = tf.nn.l2_loss((jointVec4 -jointVec9) -(jointVec4_GT -jointVec9_GT))
	compLoss63  = tf.nn.l2_loss((jointVec4 -jointVec10)-(jointVec4_GT -jointVec10_GT))
	compLoss64  = tf.nn.l2_loss((jointVec4 -jointVec11)-(jointVec4_GT -jointVec11_GT))
	compLoss65  = tf.nn.l2_loss((jointVec4 -jointVec12)-(jointVec4_GT -jointVec12_GT))
	compLoss66  = tf.nn.l2_loss((jointVec4 -jointVec13)-(jointVec4_GT -jointVec13_GT))
	compLoss67  = tf.nn.l2_loss((jointVec4 -jointVec14)-(jointVec4_GT -jointVec14_GT))
	compLoss68  = tf.nn.l2_loss((jointVec4 -jointVec15)-(jointVec4_GT -jointVec15_GT))
	compLoss69  = tf.nn.l2_loss((jointVec4 -jointVec16)-(jointVec4_GT -jointVec16_GT))
	compLoss70  = tf.nn.l2_loss((jointVec4 -jointVec17)-(jointVec4_GT -jointVec17_GT))
	compLoss71  = tf.nn.l2_loss((jointVec4 -jointVec18)-(jointVec4_GT -jointVec18_GT))
	compLoss72  = tf.nn.l2_loss((jointVec4 -jointVec19)-(jointVec4_GT -jointVec19_GT))
	compLoss73  = tf.nn.l2_loss((jointVec4 -jointVec20)-(jointVec4_GT -jointVec20_GT))
	compLoss74  = tf.nn.l2_loss((jointVec4 -jointVec21)-(jointVec4_GT -jointVec21_GT))

	compLoss75  = tf.nn.l2_loss((jointVec5 -jointVec6) -(jointVec5_GT -jointVec6_GT))
	compLoss76  = tf.nn.l2_loss((jointVec5 -jointVec7) -(jointVec5_GT -jointVec7_GT))
	compLoss77  = tf.nn.l2_loss((jointVec5 -jointVec8) -(jointVec5_GT -jointVec8_GT))
	compLoss78  = tf.nn.l2_loss((jointVec5 -jointVec9) -(jointVec5_GT -jointVec9_GT))
	compLoss79  = tf.nn.l2_loss((jointVec5 -jointVec10)-(jointVec5_GT -jointVec10_GT))
	compLoss80  = tf.nn.l2_loss((jointVec5 -jointVec11)-(jointVec5_GT -jointVec11_GT))
	compLoss81  = tf.nn.l2_loss((jointVec5 -jointVec12)-(jointVec5_GT -jointVec12_GT))
	compLoss82  = tf.nn.l2_loss((jointVec5 -jointVec13)-(jointVec5_GT -jointVec13_GT))
	compLoss83  = tf.nn.l2_loss((jointVec5 -jointVec14)-(jointVec5_GT -jointVec14_GT))
	compLoss84  = tf.nn.l2_loss((jointVec5 -jointVec15)-(jointVec5_GT -jointVec15_GT))
	compLoss85  = tf.nn.l2_loss((jointVec5 -jointVec16)-(jointVec5_GT -jointVec16_GT))
	compLoss86  = tf.nn.l2_loss((jointVec5 -jointVec17)-(jointVec5_GT -jointVec17_GT))
	compLoss87  = tf.nn.l2_loss((jointVec5 -jointVec18)-(jointVec5_GT -jointVec18_GT))
	compLoss88  = tf.nn.l2_loss((jointVec5 -jointVec19)-(jointVec5_GT -jointVec19_GT))
	compLoss89  = tf.nn.l2_loss((jointVec5 -jointVec20)-(jointVec5_GT -jointVec20_GT))
	compLoss90  = tf.nn.l2_loss((jointVec5 -jointVec21)-(jointVec5_GT -jointVec21_GT))

	compLoss91  = tf.nn.l2_loss((jointVec6 -jointVec7) -(jointVec6_GT -jointVec7_GT))
	compLoss92  = tf.nn.l2_loss((jointVec6 -jointVec8) -(jointVec6_GT -jointVec8_GT))
	compLoss93  = tf.nn.l2_loss((jointVec6 -jointVec9) -(jointVec6_GT -jointVec9_GT))
	compLoss94  = tf.nn.l2_loss((jointVec6 -jointVec10)-(jointVec6_GT -jointVec10_GT))
	compLoss95  = tf.nn.l2_loss((jointVec6 -jointVec11)-(jointVec6_GT -jointVec11_GT))
	compLoss96  = tf.nn.l2_loss((jointVec6 -jointVec12)-(jointVec6_GT -jointVec12_GT))
	compLoss97  = tf.nn.l2_loss((jointVec6 -jointVec13)-(jointVec6_GT -jointVec13_GT))
	compLoss98  = tf.nn.l2_loss((jointVec6 -jointVec14)-(jointVec6_GT -jointVec14_GT))
	compLoss99  = tf.nn.l2_loss((jointVec6 -jointVec15)-(jointVec6_GT -jointVec15_GT))
	compLoss100 = tf.nn.l2_loss((jointVec6 -jointVec16)-(jointVec6_GT -jointVec16_GT))
	compLoss101 = tf.nn.l2_loss((jointVec6 -jointVec17)-(jointVec6_GT -jointVec17_GT))
	compLoss102 = tf.nn.l2_loss((jointVec6 -jointVec18)-(jointVec6_GT -jointVec18_GT))
	compLoss103 = tf.nn.l2_loss((jointVec6 -jointVec19)-(jointVec6_GT -jointVec19_GT))
	compLoss104 = tf.nn.l2_loss((jointVec6 -jointVec20)-(jointVec6_GT -jointVec20_GT))
	compLoss105 = tf.nn.l2_loss((jointVec6 -jointVec21)-(jointVec6_GT -jointVec21_GT))

	compLoss106 = tf.nn.l2_loss((jointVec7 -jointVec8) -(jointVec7_GT -jointVec8_GT))
	compLoss107 = tf.nn.l2_loss((jointVec7 -jointVec9) -(jointVec7_GT -jointVec9_GT))
	compLoss108 = tf.nn.l2_loss((jointVec7 -jointVec10)-(jointVec7_GT -jointVec10_GT))
	compLoss109 = tf.nn.l2_loss((jointVec7 -jointVec11)-(jointVec7_GT -jointVec11_GT))
	compLoss110 = tf.nn.l2_loss((jointVec7 -jointVec12)-(jointVec7_GT -jointVec12_GT))
	compLoss111 = tf.nn.l2_loss((jointVec7 -jointVec13)-(jointVec7_GT -jointVec13_GT))
	compLoss112 = tf.nn.l2_loss((jointVec7 -jointVec14)-(jointVec7_GT -jointVec14_GT))
	compLoss113 = tf.nn.l2_loss((jointVec7 -jointVec15)-(jointVec7_GT -jointVec15_GT))
	compLoss114 = tf.nn.l2_loss((jointVec7 -jointVec16)-(jointVec7_GT -jointVec16_GT))
	compLoss115 = tf.nn.l2_loss((jointVec7 -jointVec17)-(jointVec7_GT -jointVec17_GT))
	compLoss116 = tf.nn.l2_loss((jointVec7 -jointVec18)-(jointVec7_GT -jointVec18_GT))
	compLoss117 = tf.nn.l2_loss((jointVec7 -jointVec19)-(jointVec7_GT -jointVec19_GT))
	compLoss118 = tf.nn.l2_loss((jointVec7 -jointVec20)-(jointVec7_GT -jointVec20_GT))
	compLoss119 = tf.nn.l2_loss((jointVec7 -jointVec21)-(jointVec7_GT -jointVec21_GT))

	compLoss120 = tf.nn.l2_loss((jointVec8 -jointVec9) -(jointVec8_GT -jointVec9_GT))
	compLoss121 = tf.nn.l2_loss((jointVec8 -jointVec10)-(jointVec8_GT -jointVec10_GT))
	compLoss122 = tf.nn.l2_loss((jointVec8 -jointVec11)-(jointVec8_GT -jointVec11_GT))
	compLoss123 = tf.nn.l2_loss((jointVec8 -jointVec12)-(jointVec8_GT -jointVec12_GT))
	compLoss124 = tf.nn.l2_loss((jointVec8 -jointVec13)-(jointVec8_GT -jointVec13_GT))
	compLoss125 = tf.nn.l2_loss((jointVec8 -jointVec14)-(jointVec8_GT -jointVec14_GT))
	compLoss126 = tf.nn.l2_loss((jointVec8 -jointVec15)-(jointVec8_GT -jointVec15_GT))
	compLoss127 = tf.nn.l2_loss((jointVec8 -jointVec16)-(jointVec8_GT -jointVec16_GT))
	compLoss128 = tf.nn.l2_loss((jointVec8 -jointVec17)-(jointVec8_GT -jointVec17_GT))
	compLoss129 = tf.nn.l2_loss((jointVec8 -jointVec18)-(jointVec8_GT -jointVec18_GT))
	compLoss130 = tf.nn.l2_loss((jointVec8 -jointVec19)-(jointVec8_GT -jointVec19_GT))
	compLoss131 = tf.nn.l2_loss((jointVec8 -jointVec20)-(jointVec8_GT -jointVec20_GT))
	compLoss132 = tf.nn.l2_loss((jointVec8 -jointVec21)-(jointVec8_GT -jointVec21_GT))

	compLoss133 = tf.nn.l2_loss((jointVec9 -jointVec10)-(jointVec9_GT -jointVec10_GT))
	compLoss134 = tf.nn.l2_loss((jointVec9 -jointVec11)-(jointVec9_GT -jointVec11_GT))
	compLoss135 = tf.nn.l2_loss((jointVec9 -jointVec12)-(jointVec9_GT -jointVec12_GT))
	compLoss136 = tf.nn.l2_loss((jointVec9 -jointVec13)-(jointVec9_GT -jointVec13_GT))
	compLoss137 = tf.nn.l2_loss((jointVec9 -jointVec14)-(jointVec9_GT -jointVec14_GT))
	compLoss138 = tf.nn.l2_loss((jointVec9 -jointVec15)-(jointVec9_GT -jointVec15_GT))
	compLoss139 = tf.nn.l2_loss((jointVec9 -jointVec16)-(jointVec9_GT -jointVec16_GT))
	compLoss140 = tf.nn.l2_loss((jointVec9 -jointVec17)-(jointVec9_GT -jointVec17_GT))
	compLoss141 = tf.nn.l2_loss((jointVec9 -jointVec18)-(jointVec9_GT -jointVec18_GT))
	compLoss142 = tf.nn.l2_loss((jointVec9 -jointVec19)-(jointVec9_GT -jointVec19_GT))
	compLoss143 = tf.nn.l2_loss((jointVec9 -jointVec20)-(jointVec9_GT -jointVec20_GT))
	compLoss144 = tf.nn.l2_loss((jointVec9 -jointVec21)-(jointVec9_GT -jointVec21_GT))

	compLoss145 = tf.nn.l2_loss((jointVec10-jointVec11)-(jointVec10_GT-jointVec11_GT))
	compLoss146 = tf.nn.l2_loss((jointVec10-jointVec12)-(jointVec10_GT-jointVec12_GT))
	compLoss147 = tf.nn.l2_loss((jointVec10-jointVec13)-(jointVec10_GT-jointVec13_GT))
	compLoss148 = tf.nn.l2_loss((jointVec10-jointVec14)-(jointVec10_GT-jointVec14_GT))
	compLoss149 = tf.nn.l2_loss((jointVec10-jointVec15)-(jointVec10_GT-jointVec15_GT))
	compLoss150 = tf.nn.l2_loss((jointVec10-jointVec16)-(jointVec10_GT-jointVec16_GT))
	compLoss151 = tf.nn.l2_loss((jointVec10-jointVec17)-(jointVec10_GT-jointVec17_GT))
	compLoss152 = tf.nn.l2_loss((jointVec10-jointVec18)-(jointVec10_GT-jointVec18_GT))
	compLoss153 = tf.nn.l2_loss((jointVec10-jointVec19)-(jointVec10_GT-jointVec19_GT))
	compLoss154 = tf.nn.l2_loss((jointVec10-jointVec20)-(jointVec10_GT-jointVec20_GT))
	compLoss155 = tf.nn.l2_loss((jointVec10-jointVec21)-(jointVec10_GT-jointVec21_GT))

	compLoss156 = tf.nn.l2_loss((jointVec11-jointVec12)-(jointVec11_GT-jointVec12_GT))
	compLoss157 = tf.nn.l2_loss((jointVec11-jointVec13)-(jointVec11_GT-jointVec13_GT))
	compLoss158 = tf.nn.l2_loss((jointVec11-jointVec14)-(jointVec11_GT-jointVec14_GT))
	compLoss159 = tf.nn.l2_loss((jointVec11-jointVec15)-(jointVec11_GT-jointVec15_GT))
	compLoss160 = tf.nn.l2_loss((jointVec11-jointVec16)-(jointVec11_GT-jointVec16_GT))
	compLoss161 = tf.nn.l2_loss((jointVec11-jointVec17)-(jointVec11_GT-jointVec17_GT))
	compLoss162 = tf.nn.l2_loss((jointVec11-jointVec18)-(jointVec11_GT-jointVec18_GT))
	compLoss163 = tf.nn.l2_loss((jointVec11-jointVec19)-(jointVec11_GT-jointVec19_GT))
	compLoss164 = tf.nn.l2_loss((jointVec11-jointVec20)-(jointVec11_GT-jointVec20_GT))
	compLoss165 = tf.nn.l2_loss((jointVec11-jointVec21)-(jointVec11_GT-jointVec21_GT))

	compLoss166 = tf.nn.l2_loss((jointVec12-jointVec13)-(jointVec12_GT-jointVec13_GT))
	compLoss167 = tf.nn.l2_loss((jointVec12-jointVec14)-(jointVec12_GT-jointVec14_GT))
	compLoss168 = tf.nn.l2_loss((jointVec12-jointVec15)-(jointVec12_GT-jointVec15_GT))
	compLoss169 = tf.nn.l2_loss((jointVec12-jointVec16)-(jointVec12_GT-jointVec16_GT))
	compLoss170 = tf.nn.l2_loss((jointVec12-jointVec17)-(jointVec12_GT-jointVec17_GT))
	compLoss171 = tf.nn.l2_loss((jointVec12-jointVec18)-(jointVec12_GT-jointVec18_GT))
	compLoss172 = tf.nn.l2_loss((jointVec12-jointVec19)-(jointVec12_GT-jointVec19_GT))
	compLoss173 = tf.nn.l2_loss((jointVec12-jointVec20)-(jointVec12_GT-jointVec20_GT))
	compLoss174 = tf.nn.l2_loss((jointVec12-jointVec21)-(jointVec12_GT-jointVec21_GT))

	compLoss175 = tf.nn.l2_loss((jointVec13-jointVec14)-(jointVec13_GT-jointVec14_GT))
	compLoss176 = tf.nn.l2_loss((jointVec13-jointVec15)-(jointVec13_GT-jointVec15_GT))
	compLoss177 = tf.nn.l2_loss((jointVec13-jointVec16)-(jointVec13_GT-jointVec16_GT))
	compLoss178 = tf.nn.l2_loss((jointVec13-jointVec17)-(jointVec13_GT-jointVec17_GT))
	compLoss179 = tf.nn.l2_loss((jointVec13-jointVec18)-(jointVec13_GT-jointVec18_GT))
	compLoss180 = tf.nn.l2_loss((jointVec13-jointVec19)-(jointVec13_GT-jointVec19_GT))
	compLoss181 = tf.nn.l2_loss((jointVec13-jointVec20)-(jointVec13_GT-jointVec20_GT))
	compLoss182 = tf.nn.l2_loss((jointVec13-jointVec21)-(jointVec13_GT-jointVec21_GT))

	compLoss183 = tf.nn.l2_loss((jointVec14-jointVec15)-(jointVec14_GT-jointVec15_GT))
	compLoss184 = tf.nn.l2_loss((jointVec14-jointVec16)-(jointVec14_GT-jointVec16_GT))
	compLoss185 = tf.nn.l2_loss((jointVec14-jointVec17)-(jointVec14_GT-jointVec17_GT))
	compLoss186 = tf.nn.l2_loss((jointVec14-jointVec18)-(jointVec14_GT-jointVec18_GT))
	compLoss187 = tf.nn.l2_loss((jointVec14-jointVec19)-(jointVec14_GT-jointVec19_GT))
	compLoss188 = tf.nn.l2_loss((jointVec14-jointVec20)-(jointVec14_GT-jointVec20_GT))
	compLoss189 = tf.nn.l2_loss((jointVec14-jointVec21)-(jointVec14_GT-jointVec21_GT))

	compLoss190 = tf.nn.l2_loss((jointVec15-jointVec16)-(jointVec15_GT-jointVec16_GT))
	compLoss191 = tf.nn.l2_loss((jointVec15-jointVec17)-(jointVec15_GT-jointVec17_GT))
	compLoss192 = tf.nn.l2_loss((jointVec15-jointVec18)-(jointVec15_GT-jointVec18_GT))
	compLoss193 = tf.nn.l2_loss((jointVec15-jointVec19)-(jointVec15_GT-jointVec19_GT))
	compLoss194 = tf.nn.l2_loss((jointVec15-jointVec20)-(jointVec15_GT-jointVec20_GT))
	compLoss195 = tf.nn.l2_loss((jointVec15-jointVec21)-(jointVec15_GT-jointVec21_GT))

	compLoss196 = tf.nn.l2_loss((jointVec16-jointVec17)-(jointVec16_GT-jointVec17_GT))
	compLoss197 = tf.nn.l2_loss((jointVec16-jointVec18)-(jointVec16_GT-jointVec18_GT))
	compLoss198 = tf.nn.l2_loss((jointVec16-jointVec19)-(jointVec16_GT-jointVec19_GT))
	compLoss199 = tf.nn.l2_loss((jointVec16-jointVec20)-(jointVec16_GT-jointVec20_GT))
	compLoss200 = tf.nn.l2_loss((jointVec16-jointVec21)-(jointVec16_GT-jointVec21_GT))

	compLoss201 = tf.nn.l2_loss((jointVec17-jointVec18)-(jointVec17_GT-jointVec18_GT))
	compLoss202 = tf.nn.l2_loss((jointVec17-jointVec19)-(jointVec17_GT-jointVec19_GT))
	compLoss203 = tf.nn.l2_loss((jointVec17-jointVec20)-(jointVec17_GT-jointVec20_GT))
	compLoss204 = tf.nn.l2_loss((jointVec17-jointVec21)-(jointVec17_GT-jointVec21_GT))

	compLoss205 = tf.nn.l2_loss((jointVec18-jointVec19)-(jointVec18_GT-jointVec19_GT))
	compLoss206 = tf.nn.l2_loss((jointVec18-jointVec20)-(jointVec18_GT-jointVec20_GT))
	compLoss207 = tf.nn.l2_loss((jointVec18-jointVec21)-(jointVec18_GT-jointVec21_GT))

	compLoss208 = tf.nn.l2_loss((jointVec19-jointVec20)-(jointVec19_GT-jointVec20_GT))
	compLoss209 = tf.nn.l2_loss((jointVec19-jointVec21)-(jointVec19_GT-jointVec21_GT))

	compLoss210 = tf.nn.l2_loss((jointVec20-jointVec21)-(jointVec20_GT-jointVec21_GT))

	compLoss = 	(compLoss1  +compLoss2  +compLoss3  +compLoss4  +compLoss5  +compLoss6  +compLoss7  +compLoss8  +compLoss9  +compLoss10 
				+compLoss11 +compLoss12 +compLoss13 +compLoss14 +compLoss15 +compLoss16 +compLoss17 +compLoss18 +compLoss19 +compLoss20
				+compLoss21 +compLoss22 +compLoss23 +compLoss24 +compLoss25 +compLoss26 +compLoss27 +compLoss28 +compLoss29 +compLoss30
				+compLoss31 +compLoss32 +compLoss33 +compLoss34 +compLoss35 +compLoss36 +compLoss37 +compLoss38 +compLoss39 +compLoss40
				+compLoss41 +compLoss42 +compLoss43 +compLoss44 +compLoss45 +compLoss46 +compLoss47 +compLoss48 +compLoss49 +compLoss50
				+compLoss51 +compLoss52 +compLoss53 +compLoss54 +compLoss55 +compLoss56 +compLoss57 +compLoss58 +compLoss59 +compLoss60
				+compLoss61 +compLoss62 +compLoss63 +compLoss64 +compLoss65 +compLoss66 +compLoss67 +compLoss68 +compLoss69 +compLoss70
				+compLoss71 +compLoss72 +compLoss73 +compLoss74 +compLoss75 +compLoss76 +compLoss77 +compLoss78 +compLoss79 +compLoss80
				+compLoss81 +compLoss82 +compLoss83 +compLoss84 +compLoss85 +compLoss86 +compLoss87 +compLoss88 +compLoss89 +compLoss90
				+compLoss91 +compLoss92 +compLoss93 +compLoss94 +compLoss95 +compLoss96 +compLoss97 +compLoss98 +compLoss99 +compLoss100
				+compLoss101+compLoss102+compLoss103+compLoss104+compLoss105+compLoss106+compLoss107+compLoss108+compLoss109+compLoss110
				+compLoss111+compLoss112+compLoss113+compLoss114+compLoss115+compLoss116+compLoss117+compLoss118+compLoss119+compLoss120
				+compLoss121+compLoss122+compLoss123+compLoss124+compLoss125+compLoss126+compLoss127+compLoss128+compLoss129+compLoss130
				+compLoss131+compLoss132+compLoss133+compLoss134+compLoss135+compLoss136+compLoss137+compLoss138+compLoss139+compLoss140
				+compLoss141+compLoss142+compLoss143+compLoss144+compLoss145+compLoss146+compLoss147+compLoss148+compLoss149+compLoss150
				+compLoss151+compLoss152+compLoss153+compLoss154+compLoss155+compLoss156+compLoss157+compLoss158+compLoss159+compLoss160
				+compLoss161+compLoss162+compLoss163+compLoss164+compLoss165+compLoss166+compLoss167+compLoss168+compLoss169+compLoss170
				+compLoss171+compLoss172+compLoss173+compLoss174+compLoss175+compLoss176+compLoss177+compLoss178+compLoss179+compLoss180
				+compLoss181+compLoss182+compLoss183+compLoss184+compLoss185+compLoss186+compLoss187+compLoss188+compLoss189+compLoss190
				+compLoss191+compLoss192+compLoss193+compLoss194+compLoss195+compLoss196+compLoss197+compLoss198+compLoss199+compLoss200
				+compLoss201+compLoss202+compLoss203+compLoss204+compLoss205+compLoss206+compLoss207+compLoss208+compLoss209+compLoss210)

	# Combine individual losses (weighted)
	combined_loss = poseLoss + (1/210)*compLoss

	return combined_loss


def train_posenet(loss, learning_rate):
	"""Sets up the training operations.
	Creates a summarizer to track the loss and the learning rate over time in TensorBoard.
	Creates an optimizer and applies the gradients to all trainable variables.
	The operation returned by this function is what must be passed to the sess.run() call to cause the model to train.
	Args:
		loss: Loss tensor, from loss().
		learning_rate: The learning rate to be used for Adam.
	Returns:
		train_posenet_op: The Op for training.
	"""
	
	# Add a scalar summary for the snapshot loss.
	tf.summary.scalar('loss_posenet', loss)
	tf.summary.scalar('learning_rate_posenet', learning_rate)
	
	# Create the Adam optimizer with the given learning rate.
	optimizer = tf.train.AdamOptimizer(learning_rate)
	
	# Create a variable to track the global step.
	global_step = tf.Variable(0, name='global_step', trainable=False)
	
	# Use the optimizer to apply the gradients that minimize the loss (and also increment the global step counter) as a single training step.
	train_posenet_op = optimizer.minimize(loss, global_step=global_step)
	
	return train_posenet_op
