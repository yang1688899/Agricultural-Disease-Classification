import tensorflow as tf
import tensorflow.keras as k
from tensorflow.layers import flatten

def weights_variable(shape, stddev=0.1, name=None):
    initial = tf.truncated_normal(shape=shape,stddev=stddev)
    return tf.Variable(initial,name=name)

def bias_variable(shape,bias=0.1, name=None):
     initial = tf.constant(bias,shape=shape)
     return tf.Variable(initial, name=None)

def network(x,rate):
    vgg = k.applications.VGG19(include_top=False,input_tensor=x)
    vgg_notop = vgg.output

    with tf.name_scope("buttom"):
        flat = flatten(vgg_notop)

        fc1_w = weights_variable([25088,1024],name="fc1_w")
        fc1_b = bias_variable([1024],name="fc1_b")
        fc1 = tf.nn.relu( tf.matmul(flat,fc1_w)+fc1_b, name="fc1" )
        drop_fc1 = tf.nn.dropout(fc1,keep_prob=rate)

        fc2_w = weights_variable([1024,512],name="fc2_w")
        fc2_b = bias_variable([512],name="fc2_b")
        fc2 = tf.nn.relu( tf.matmul(drop_fc1,fc2_w)+fc2_b, name="fc2" )
        drop_fc2 = tf.nn.dropout(fc2, keep_prob=rate)

        fc3_w = weights_variable([512,61],name="fc3_w")
        fc3_b = weights_variable([61], name="fc3_b")
        logit = tf.add( tf.matmul(drop_fc2,fc3_w),fc3_b, name="logit")

    return logit
