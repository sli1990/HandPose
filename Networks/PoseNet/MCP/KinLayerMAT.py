# Python 2 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import
import time
import math
import tensorflow as tf
from utilities import DH_matrix_mul, ROT_X, ROT_Y, ROT_Z, TRANSL, HOM_TRANS

def KinLayer(input):

    mcpBase = input[0] 
    fingerVectors = input[1]
    jointAngles = input[2]
    boneLengths = input[3]

    # DH PARAMETER LIST (theta,d,r,alpha)
    DH_PARAMETERS = tf.Variable([[0, 0, 0, -math.pi/2],                     # T_PIP_1  
                                 [math.pi/6, 0, boneLengths[5], math.pi/2], # T_PIP_2
                                 [0, 0, 0, -math.pi/2],                     # T_DIP_1
                                 [0, 0, boneLengths[6], 0],                 # T_DIP_2
                                 [0, 0, boneLengths[7], 0],                 # T_TIP
                                 [math.pi, 0, 0, -math.pi/2],               # I_PIP_1
                                 [-math.pi/2, 0, 0, math.pi/2],             # I_PIP_2
                                 [0, 0, boneLengths[8], 0],                 # I_PIP_3
                                 [0, 0, boneLengths[9], 0],                 # I_DIP
                                 [0, 0, boneLengths[10], 0],                # I_TIP
                                 [math.pi, 0, 0, -math.pi/2],               # M_PIP_1
                                 [-math.pi/2, 0, 0, math.pi/2],             # M_PIP_2
                                 [0, 0, boneLengths[11], 0],                # M_PIP_3
                                 [0, 0, boneLengths[12], 0],                # M_DIP
                                 [0, 0, boneLengths[13], 0],                # M_TIP
                                 [math.pi, 0, 0, -math.pi/2],               # R_PIP_1
                                 [-math.pi/2, 0, 0, math.pi/2],             # R_PIP_2
                                 [0, 0, boneLengths[14], 0],                # R_PIP_3
                                 [0, 0, boneLengths[15], 0],                # R_DIP
                                 [0, 0, boneLengths[16], 0],                # R_TIP
                                 [math.pi, 0, 0, -math.pi/2],               # P_PIP_1
                                 [-math.pi/2, 0, 0, math.pi/2],             # P_PIP_2
                                 [0, 0, boneLengths[17], 0],                # P_PIP_3
                                 [0, 0, boneLengths[18], 0],                # P_DIP
                                 [0, 0, boneLengths[19], 0]])               # P_TIP
                                                

    # Mean values of finger rotation around wrist base axis
    fingerAnglesMean = tf.constant([-0.0872665, 0.0, 0.0872665, 0.0872665])

    # MCP (MIDDLE BASE)
    MAT_MCP_BASE = HOM_TRANS(mcpBase[5],mcpBase[4],mcpBase[3],mcpBase[0],mcpBase[1],mcpBase[2])

    # WRIST 
    MAT_WRIST = tf.matmul(MAT_MCP_BASE,TRANSL(fingerVectors[0],fingerVectors[1],fingerVectors[2]))
    WRIST = MAT_WRIST[0:3,3] 

    # THUMB
    MAT_T_MCP = tf.matmul(MAT_MCP_BASE,TRANSL(fingerVectors[3],fingerVectors[4],fingerVectors[5]))
    MAT_T_MCP = tf.matmul(MAT_T_MCP,ROT_X(math.pi))
    T_MCP = MAT_T_MCP[0:3,3]

    MAT_T_PIP = tf.matmul(MAT_T_MCP,DH_matrix_mul(jointAngles[0],0,DH_PARAMETERS[0][2],DH_PARAMETERS[0][3]))
    MAT_T_PIP = tf.matmul(MAT_T_PIP,DH_matrix_mul(jointAngles[1]+DH_PARAMETERS[1][0],0,DH_PARAMETERS[1][2],DH_PARAMETERS[1][3]))
    T_PIP = MAT_T_PIP[0:3,3]

    MAT_T_DIP = tf.matmul(MAT_T_PIP,DH_matrix_mul(jointAngles[2],0,DH_PARAMETERS[2][2],DH_PARAMETERS[2][3]))
    MAT_T_DIP = tf.matmul(MAT_T_DIP,DH_matrix_mul(jointAngles[3],0,DH_PARAMETERS[3][2],DH_PARAMETERS[3][3]))
    T_DIP = MAT_T_DIP[0:3,3]

    MAT_T_TIP = tf.matmul(MAT_T_DIP,DH_matrix_mul(jointAngles[4],0,DH_PARAMETERS[4][2],DH_PARAMETERS[4][3]))
    T_TIP = MAT_T_TIP[0:3,3]    

    # INDEX
    MAT_I_MCP = tf.matmul(MAT_MCP_BASE,TRANSL(fingerVectors[6],fingerVectors[7],fingerVectors[8]))
    MAT_I_MCP = tf.matmul(MAT_I_MCP,ROT_Z(fingerAnglesMean[0]))
    I_MCP = MAT_I_MCP[0:3,3]

    MAT_I_PIP = tf.matmul(MAT_I_MCP,DH_matrix_mul(jointAngles[5]+DH_PARAMETERS[5][0],0,DH_PARAMETERS[5][2],DH_PARAMETERS[5][3]))
    MAT_I_PIP = tf.matmul(MAT_I_PIP,DH_matrix_mul(jointAngles[6]+DH_PARAMETERS[6][0],0,DH_PARAMETERS[6][2],DH_PARAMETERS[6][3]))
    MAT_I_PIP = tf.matmul(MAT_I_PIP,DH_matrix_mul(jointAngles[7],0,DH_PARAMETERS[7][2],DH_PARAMETERS[7][3]))
    I_PIP = MAT_I_PIP[0:3,3]

    MAT_I_DIP = tf.matmul(MAT_I_PIP,DH_matrix_mul(jointAngles[8],0,DH_PARAMETERS[8][2],DH_PARAMETERS[8][3]))
    I_DIP = MAT_I_DIP[0:3,3]

    MAT_I_TIP = tf.matmul(MAT_I_DIP,DH_matrix_mul(jointAngles[9],0,DH_PARAMETERS[9][2],DH_PARAMETERS[9][3]))
    I_TIP = MAT_I_TIP[0:3,3] 

    # MIDDLE
    MAT_M_MCP = MAT_MCP_BASE
    M_MCP = MAT_M_MCP[0:3,3]

    MAT_M_PIP = tf.matmul(MAT_M_MCP,DH_matrix_mul(jointAngles[10]+DH_PARAMETERS[10][0],0,DH_PARAMETERS[10][2],DH_PARAMETERS[10][3]))
    MAT_M_PIP = tf.matmul(MAT_M_PIP,DH_matrix_mul(jointAngles[11]+DH_PARAMETERS[11][0],0,DH_PARAMETERS[11][2],DH_PARAMETERS[11][3]))
    MAT_M_PIP = tf.matmul(MAT_M_PIP,DH_matrix_mul(jointAngles[12],0,DH_PARAMETERS[12][2],DH_PARAMETERS[12][3]))
    M_PIP = MAT_M_PIP[0:3,3]

    MAT_M_DIP = tf.matmul(MAT_M_PIP,DH_matrix_mul(jointAngles[13],0,DH_PARAMETERS[13][2],DH_PARAMETERS[13][3]))
    M_DIP = MAT_M_DIP[0:3,3]

    MAT_M_TIP = tf.matmul(MAT_M_DIP,DH_matrix_mul(jointAngles[14],0,DH_PARAMETERS[14][2],DH_PARAMETERS[14][3]))
    M_TIP = MAT_M_TIP[0:3,3]

    # RING
    MAT_R_MCP = tf.matmul(MAT_MCP_BASE,TRANSL(fingerVectors[9],fingerVectors[10],fingerVectors[11]))
    MAT_R_MCP = tf.matmul(MAT_R_MCP,ROT_Z(fingerAnglesMean[2]))
    R_MCP = MAT_R_MCP[0:3,3]

    MAT_R_PIP = tf.matmul(MAT_R_MCP,DH_matrix_mul(jointAngles[15]+DH_PARAMETERS[15][0],0,DH_PARAMETERS[15][2],DH_PARAMETERS[15][3]))
    MAT_R_PIP = tf.matmul(MAT_R_PIP,DH_matrix_mul(jointAngles[16]+DH_PARAMETERS[16][0],0,DH_PARAMETERS[16][2],DH_PARAMETERS[16][3]))
    MAT_R_PIP = tf.matmul(MAT_R_PIP,DH_matrix_mul(jointAngles[17],0,DH_PARAMETERS[17][2],DH_PARAMETERS[17][3]))
    R_PIP = MAT_R_PIP[0:3,3]

    MAT_R_DIP = tf.matmul(MAT_R_PIP,DH_matrix_mul(jointAngles[18],0,DH_PARAMETERS[18][2],DH_PARAMETERS[18][3]))
    R_DIP = MAT_R_DIP[0:3,3]

    MAT_R_TIP = tf.matmul(MAT_R_DIP,DH_matrix_mul(jointAngles[19],0,DH_PARAMETERS[19][2],DH_PARAMETERS[19][3]))
    R_TIP = MAT_R_TIP[0:3,3]

    # PINKY
    MAT_P_MCP = tf.matmul(MAT_MCP_BASE,TRANSL(fingerVectors[12],fingerVectors[13],fingerVectors[14]))
    MAT_P_MCP = tf.matmul(MAT_P_MCP,ROT_Z(fingerAnglesMean[3]))
    P_MCP = MAT_P_MCP[0:3,3]

    MAT_P_PIP = tf.matmul(MAT_P_MCP,DH_matrix_mul(jointAngles[20]+DH_PARAMETERS[20][0],0,DH_PARAMETERS[20][2],DH_PARAMETERS[20][3]))
    MAT_P_PIP = tf.matmul(MAT_P_PIP,DH_matrix_mul(jointAngles[21]+DH_PARAMETERS[21][0],0,DH_PARAMETERS[21][2],DH_PARAMETERS[21][3]))
    MAT_P_PIP = tf.matmul(MAT_P_PIP,DH_matrix_mul(jointAngles[22],0,DH_PARAMETERS[22][2],DH_PARAMETERS[22][3]))
    P_PIP = MAT_P_PIP[0:3,3]

    MAT_P_DIP = tf.matmul(MAT_P_PIP,DH_matrix_mul(jointAngles[23],0,DH_PARAMETERS[23][2],DH_PARAMETERS[23][3]))
    P_DIP = MAT_P_DIP[0:3,3]

    MAT_P_TIP = tf.matmul(MAT_P_DIP,DH_matrix_mul(jointAngles[24],0,DH_PARAMETERS[24][2],DH_PARAMETERS[24][3]))
    P_TIP = MAT_P_TIP[0:3,3]

    return tf.concat([WRIST,T_MCP,I_MCP,M_MCP,R_MCP,P_MCP,T_PIP,T_DIP,T_TIP,I_PIP,I_DIP,I_TIP,M_PIP,M_DIP,M_TIP,R_PIP,R_DIP,R_TIP,P_PIP,P_DIP,P_TIP],0)

def PCA_Layer(boneShapes):

    # PCA transformation matrix and mean values for bone lenghts
    bonePCA = tf.constant([ [ 0.0796519, -0.0101939,  0.0240200,  0.0060815,  0.0011960, -0.0067941,  0.0075579, -0.0221086,  0.0095106],
                            [ 0.3327169,  0.0780853,  0.1679594,  0.0506008, -0.2845224,  0.1445072,  0.0032098, -0.1867533,  0.7000818],
                            [ 0.2841618, -0.0163130,  0.2667167, -0.0038051,  0.1078237, -0.1547226,  0.0928975, -0.1575557, -0.3446003],
                            [ 0.4102765, -0.1070683,  0.0594828,  0.0349966,  0.0287502, -0.0258785,  0.0209484, -0.0844844,  0.0943907],
                            [ 0.5184696, -0.1525144, -0.0280565,  0.0362169,  0.1711556, -0.0957437,  0.0296043, -0.0002186, -0.2653203],
                            [-0.4080681, -0.2146734,  0.6228724, -0.1900152,  0.2105425, -0.2423696,  0.1739814,  0.1034112,  0.1854910],
                            [ 0.1096061,  0.2086404, -0.0624692,  0.1219440,  0.1468602,  0.0130518,  0.2281359,  0.8551685,  0.1334762],
                            [ 0.0218688,  0.3397699,  0.1382673, -0.0622707, -0.1338119,  0.0138821,  0.0294744,  0.0080722, -0.3993855],
                            [-0.2047711,  0.2197664,  0.2741258,  0.2285361, -0.1437550,  0.4053025,  0.1008892, -0.1300160, -0.1480252],
                            [-0.1509989, -0.3961045, -0.0871700,  0.3984311,  0.2012820,  0.6107645,  0.0294996, -0.0157609, -0.0781955],
                            [-0.0340265,  0.0421038, -0.1738292, -0.3169981, -0.2237058,  0.1865418, -0.2377925,  0.1506637, -0.0060510],
                            [-0.1086596, -0.0203877,  0.0046726,  0.2271706, -0.3535422, -0.2235604, -0.1435203, -0.0086833,  0.0365453],
                            [-0.1848448, -0.1917156, -0.1633135,  0.4903615, -0.3216479, -0.4508476, -0.1722506,  0.0738265, -0.0835450],
                            [-0.0493063, -0.1858831, -0.0945876, -0.3146712, -0.0629196,  0.0736685, -0.2122128,  0.0591693, -0.0061184],
                            [-0.1446662,  0.2446821, -0.0634175, -0.1081419, -0.1322918, -0.0545079,  0.4184934, -0.2219076, -0.0582011],
                            [-0.1322991, -0.0949799, -0.5381195, -0.1416002,  0.0462727, -0.1020022,  0.5729420, -0.1784026,  0.0997670],
                            [-0.0294221, -0.2307858, -0.0011518, -0.2227483,  0.0169985, -0.0004377, -0.1337664, -0.0216962,  0.0178221],
                            [-0.1074432,  0.4938746, -0.1280326, -0.0811735,  0.0850957,  0.0660441, -0.3031266, -0.0335557,  0.0030238],
                            [-0.1164570,  0.2302720, -0.1437393,  0.1860426,  0.6500730, -0.1910825, -0.2926905, -0.2380458,  0.2029061],
                            [-0.0857885, -0.2365746, -0.0742302, -0.3389574, -0.0098532,  0.0341842, -0.2122741,  0.0488773, -0.0935724]])
    boneMean = tf.constant([0.0142554, 0.0749878, 0.0698093, 0.0653222, 0.0665022, 0.0539771, 0.0354403, 0.0285090, 0.0437238, 0.0259233, 0.0200164, 0.0481373, 0.0297237, 0.0218163, 0.0446516, 0.0271620, 0.0220742, 0.0358953, 0.0215205, 0.0195835])

    # Transform the inputs
    boneLengths = bonePCA*boneShapes + boneMean

    return boneLengths

def ScaleLayer(input):

    joints = input[0]
    scale = input[1]

    return joints*scale