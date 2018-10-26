from keras.models import load_model
from keras import optimizers
from keras.utils import  to_categorical
import numpy as np

import network
import config
import utils


trian_img_paths,train_labels = utils.process_annotation(config.TRAIN_ANNOTATION_FILE,config.TRAIN_DIR)
train_labels = to_categorical(train_labels,num_classes=61)
train_gen = utils.train_generator(trian_img_paths,train_labels,config.BATCH_SIZE)

features,labels = next(train_gen)

print(features.shape,labels.shape)

# base_model,model = network.vgg_network()
#
# model.summary()
#
# #freeze the convolutional layers
# for layer in base_model.layers:
#     layer.trainable = False
#
# print("begin tuning......")
# print("train samples: %s"%len(train_labels))
# print("valid samples: %s"%len(valid_labels))
#
# sgd = optimizers.SGD(lr=0.001)
# model.compile(optimizer=sgd, loss='categorical_crossentropy')
# model.fit_generator(train_gen, steps_per_epoch=ceil(num_train/batch_size), epochs=epochs,validation_data=valid_gen, validation_steps=ceil(num_validation/batch_size))
#
# model.save(save_path)
#
# print("model saved")