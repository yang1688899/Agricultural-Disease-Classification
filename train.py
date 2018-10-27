from keras.models import load_model
from keras import optimizers
from keras.utils import  to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from math import ceil

import network
import config
import utils

#getting data
trian_img_paths,train_labels = utils.process_annotation(config.TRAIN_ANNOTATION_FILE,config.TRAIN_DIR)
train_labels = to_categorical(train_labels,num_classes=61)
train_gen = utils.data_generator(trian_img_paths,train_labels,config.BATCH_SIZE)

val_img_paths,val_labels = utils.process_annotation(config.VAL_ANNOTATION_FILE,config.VAL_DIR)
val_labels = to_categorical(val_labels,num_classes=61)
val_gen = utils.data_generator(val_img_paths,val_labels,config.BATCH_SIZE,is_shuffle=False)

#build model
print("the model will be save as %s"%config.SAVE_MODEL_PATH)
base_model,model = network.vgg_network()

#freeze the convolutional layers
for layer in base_model.layers:
    layer.trainable = False

model.summary()

print("begin tuning......")
print("train samples: %s"%len(train_labels))
print("valid samples: %s"%len(val_labels))

sgd = optimizers.SGD(lr=0.01)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

#callback
early_stopping = EarlyStopping(patience=3, verbose=1)
model_checkpoint = ModelCheckpoint(config.SAVE_MODEL_PATH, save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=2, min_lr=0.00001, verbose=1)

epochs = 5

history = model.fit_generator(train_gen,
                    steps_per_epoch=ceil(len(train_labels)/config.BATCH_SIZE),
                    epochs=epochs,
                    validation_data=val_gen,
                    validation_steps=ceil(len(val_labels)/config.BATCH_SIZE),
                    callbacks=[early_stopping, model_checkpoint, reduce_lr])

#load the finetuned model
model = load_model(config.SAVE_MODEL_PATH)
#unfeeze all layer
for layer in model.layers:
    layer.trainable = True

adam = optimizers.Adam(lr=0.001)
model.compile(optimizer=adam, loss='categorical_crossentropy')

#callback
early_stopping = EarlyStopping(patience=5, verbose=1)
model_checkpoint = ModelCheckpoint(config.SAVE_MODEL_PATH, save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.00001, verbose=1)

epochs = 20

history = model.fit_generator(train_gen,
                    steps_per_epoch=ceil(len(train_labels)/config.BATCH_SIZE),
                    epochs=epochs,
                    validation_data=val_gen,
                    validation_steps=ceil(len(val_labels)/config.BATCH_SIZE),
                    callbacks=[early_stopping, model_checkpoint, reduce_lr])