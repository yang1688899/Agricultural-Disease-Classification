import json
from sklearn.utils import shuffle
import cv2
import random
import numpy as np
from math import ceil
import tensorflow as tf
import pickle

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

#等比例缩放图片,size为最边短边长
def resize_img(img,size):
    h = img.shape[0]
    w = img.shape[1]
    scale = max(size/h,size/w)
    resized_img = cv2.resize( img, (int(h*scale),int(w*scale)) )
    return resized_img

#对缩放图片进行随机切割,要求输入图片其中一边与切割大小相等
def random_crop(img,size):
    h = img.shape[0]
    w = img.shape[1]
    if h>w:
        offset = random.randint(0, h - size)
        croped_img = img[offset:offset+size, :]
    else:
        offset = random.randint(0, w - size)
        croped_img = img[:, offset:offset+size]
    return croped_img

def train_agumetation(img):
    img = agumetation.resize_img(img,config.INPUT_SIZE)
    img = agumetation.random_crop(img)
    img = agumetation.random_flip(img,0.5)

    return img-128/128.

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

def data_generator(img_paths,labels,batch_size,is_shuffle=True):
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
