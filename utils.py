import json
from sklearn.utils import shuffle
import cv2
import random
import numpy as np
from math import ceil
import tensorflow as tf
import pickle
import json

import config
import agumetation

#初始化sess,或回复保存的sess
def start_or_restore_training(sess,saver,checkpoint_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print('Restore the model from checkpoint %s' % ckpt.model_checkpoint_path)
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    else:
        sess.run(tf.global_variables_initializer())
        step = 1
        print('start training from new state')
    return sess,step

def dump_to_json(path,obj):
    with open(path,"w") as f:
        json.dump(obj,f,ensure_ascii=False)

def save_to_pickle(obj,savepath):
    with open(savepath,"wb") as file:
        pickle.dump(obj,file)

def load_pickle(path):
    with open(path,"rb") as file:
        obj = pickle.load(file)
        return obj

#含中文路径读取图片方法
def cv_imread(filepath):
    img = cv2.imdecode(np.fromfile(filepath,dtype=np.uint8),-1)
    return img

def train_agumetation(img):
    img = agumetation.resize_img(img,config.INPUT_SIZE)
    img = agumetation.random_crop(img)
    img = agumetation.random_flip(img,0.5)
    img = agumetation.random_light(img)

    return (img-128.)/128.

def val_agumetation(img):
    img = agumetation.resize_img(img, config.INPUT_SIZE)
    img = agumetation.random_crop(img)
    img = agumetation.random_flip(img, 0.5)

    return (img - 128.) / 128.

# def val_agumetation(img,n_fold=3):
#     img = agumetation.resize_img(img,config.INPUT_SIZE)
#     img = agumetation.random_flip(img, 0.5)
#     imgs = agumetation.n_fold_crop(img,n_fold=3)
#     return imgs

def process_annotation(anno_file,dir):
    with open(anno_file) as file:
        annotations = json.load(file)
        img_paths = []
        labels = []
        for anno in annotations:
            img_paths.append(dir + anno["image_id"])
            labels.append(anno["disease_class"])
    return img_paths, labels

def train_generator(img_paths,labels,batch_size,is_shuffle=True):
    if is_shuffle:
        img_paths,labels = shuffle(img_paths,labels)
    num_sample = len(img_paths)
    while True:
        if is_shuffle:
            img_paths, labels = shuffle(img_paths, labels)

        for offset in range(0,num_sample,batch_size):
            batch_paths = img_paths[offset:offset+batch_size]
            batch_labels = labels[offset:offset+batch_size]

            batch_features = [train_agumetation(cv_imread(path)) for path in batch_paths]

            yield np.array(batch_features), np.array(batch_labels)

def val_generator(img_paths,labels,batch_size,is_shuffle=True):
    if is_shuffle:
        img_paths,labels = shuffle(img_paths,labels)
    num_sample = len(img_paths)
    while True:
        if is_shuffle:
            img_paths, labels = shuffle(img_paths, labels)

        for offset in range(0,num_sample,batch_size):
            batch_paths = img_paths[offset:offset+batch_size]
            batch_labels = labels[offset:offset+batch_size]

            batch_features = [val_agumetation(cv_imread(path)) for path in batch_paths]

            yield np.array(batch_features), np.array(batch_labels)

def n_crop_and_flip(img,n_crop=3):
    img = agumetation.resize_img(img,config.INPUT_SIZE)
    imgs = agumetation.n_fold_crop(img,n_crop)
    if not len(imgs)==n_crop:
        raise Exception("the number of img is not equal to n_crop!")
    imgs_flip_d = [cv2.flip(pic,1) for pic in imgs]
    imgs_flip_h1 = [cv2.flip(pic,0) for pic in imgs]
    imgs_flip_h2 = [cv2.flip(pic,0) for pic in imgs_flip_d]
    imgs.extend(imgs_flip_d)
    imgs.extend(imgs_flip_h1)
    imgs.extend(imgs_flip_h2)
    return imgs

def test_generator(paths,batch_size):
    while True:
        for offset in range(0, len(paths), batch_size):
            batch_paths = paths[offset:offset + batch_size]
            batch_imgs = [cv_imread(path) for path in batch_paths]

            batch_features = []
            for img in batch_imgs:
                imgs = n_crop_and_flip(img)
                batch_features.extend(imgs)
            batch_features  = (np.array(batch_features)-128.)/128.
            yield batch_features


# def val_generator(img_paths,labels,batch_size,n_fold=3):
#     num_sample = len(img_paths)
#     while True:
#         for offset in range(0, num_sample, batch_size):
#             batch_paths = img_paths[offset:offset + batch_size]
#             batch_labels = labels[offset:offset + batch_size]
#
#             batch_features = [train_agumetation(cv_imread(path)) for path in batch_paths]
#
#             yield np.array(batch_features), np.array(batch_labels)

def average_prediction_multi(predictions_list,num_test,n_crop=3,is_flip=True):
    if is_flip:
        num = n_crop*4
    else:
        num = n_crop
    pre_labels = []
    predictions_combine = np.sum(np.array(predictions_list),axis=0)
    for i in range(0, len(predictions_combine), num):
        batch_preds = predictions_combine[i:i + num]
        sum_pred = np.sum(batch_preds, axis=0)
        pre_labels.append(np.argmax(sum_pred))
    return pre_labels

