import network
import config
import utils

import tensorflow as tf
import os

x = tf.placeholder(dtype=tf.float32,shape=[None,224,224,3])
y = tf.placeholder(dtype=tf.int32,shape=[None,1])
one_hot_y = tf.one_hot(y,61)
rate = tf.placeholder(dtype=tf.float32)
logit = network.network(x,rate)

loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=logit,labels=one_hot_y), name="loss")
correct_pred = tf.equal( tf.cast( tf.argmax(logit,axis=1),tf.int32 ),y)
acc = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

output_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='buttom')
train_step_tune = tf.train.AdamOptimizer().minimize(loss=loss,var_list = output_vars)

train_step = tf.train.AdamOptimizer().minimize(loss)

trian_img_paths,train_labels = utils.process_annotation(config.TRAIN_ANNOTATION_FILE,config.TRAIN_DIR)
train_data_gen = utils.data_generator(trian_img_paths,train_labels,batch_size=16)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess, step = utils.start_or_restore_training(sess, saver, config.CHECKDIR)

    print("training")
    while True:
        batch_features, batch_labels = next(train_data_gen)
        if step < 5000:
            sess.run(train_step_tune, feed_dict={x:batch_features, y:batch_labels, rate:0.5})
        else:
            sess.run(train_step, feed_dict={x:batch_features, y:batch_labels, rate:0.5})

        if step%100 == 0:
            train_loss = sess.run(loss, feed_dict={x:batch_features, y:batch_labels, rate:1.})
            accuracy,val_loss = utils.validation(sess, acc, loss, x, y, rate, config.VAL_ANNOTATION_FILE, config.VAL_DIR, 16)

            print("step %s: the training loss is %s, validation loss is %s, validation accuracy is %s"%(step, train_loss, val_loss, accuracy))

        if step%1000 == 0:
            if not os.path.exists(config.CHECKDIR):
                os.mkdir(config.CHECKDIR)
            saver.save(sess, config.CHECKFILE, global_step=step)
            print('writing checkpoint at step %s' % step)

        step += 1