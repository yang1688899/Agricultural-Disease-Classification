from keras.models import load_model
from keras import optimizers
from keras.utils import  to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from math import ceil

# import config
# import utils
#
# val_img_paths,val_labels_flat = utils.process_annotation(config.VAL_ANNOTATION_FILE,config.VAL_DIR)
# val_labels = to_categorical(val_labels_flat,num_classes=61)
# val_gen = utils.val_generator(val_img_paths,val_labels,config.BATCH_SIZE,is_shuffle=False)
#
# #load model
# model = load_model(config.SAVE_MODEL_PATH)
#
# val_predictions = model.predict_generator(val_gen,
#                         steps=ceil(len(val_labels) / config.BATCH_SIZE))
#
# print(len(val_predictions))
# print(len(val_labels_flat))
#
# val_pred_label = np.argmax(val_predictions,axis=1)
# print(val_pred_label.shape)
#
# val_labels_flat = np.array(val_labels_flat)
# print(val_labels_flat.shape)
#
# print(sum(np.round(val_pred_label==val_labels_flat))/len(val_labels_flat))


import glob
from keras.models import load_model
from math import ceil
import numpy as np

import config
import utils

import cv2

def average_prediction(predictions,n_crop=3,is_flip=True):
    if is_flip:
        num = n_crop*4
    else:
        num = n_crop
    pre_labels = []
    for i in range(0, len(predictions), num):
        batch_preds = predictions[i:i + num]
        sum_pred = np.sum(batch_preds,axis=0)
        pre_labels.append(np.argmax(sum_pred))
    return pre_labels

#load model
model = load_model(config.SAVE_MODEL_PATH)

batch_size =3

val_paths = glob.glob(config.VAL_DIR+"*.jpg")
val_ids = [path.split("\\")[-1] for path in val_paths]
val_test_gen = utils.test_generator(val_paths,batch_size)

predictions = model.predict_generator(val_test_gen,
                        steps=ceil(len(val_ids) / batch_size))

print(len(predictions)/len(val_ids))

pred_labels = average_prediction(predictions,n_crop=3,is_flip=True)

pred_list = [{"image_id":id,"disease_class":int(label)} for id,label in zip(val_ids,pred_labels)]

utils.dump_to_json("./val_interceptionV3.json",pred_list)


# val_gen = utils.val_generator(val_paths,val_ids,batch_size,is_shuffle=False)
#
# for i in range(100):
#     feature_t = next(val_test_gen)
#     feature_v,_ = next(val_gen)
#     predict_t = model.predict(feature_t)
#     predict_v  = model.predict(feature_v)
#
#     print(type(feature_t[0][0][0][0]))
#     print(type(feature_v[0][0][0][0]))
#
#     print(np.argmax(predict_t,axis=1))
#     print(np.argmax(predict_v))
#
#