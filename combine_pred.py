import numpy as np
import glob

import utils
import config

def average_prediction(predictions, n_crop=3, is_flip=True):
    if is_flip:
        num = n_crop * 4
    else:
        num = n_crop
    pre_labels = []
    for i in range(0, len(predictions), num):
        batch_preds = predictions[i:i + num]
        sum_pred = np.sum(batch_preds, axis=0)
        pre_labels.append(np.argmax(sum_pred))
    return pre_labels


folds = 5
n_crop = 5
batch_size = 2

test_paths = glob.glob(config.TEST_DIR_B+"*.jpg")
test_ids = [path.split("\\")[-1] for path in test_paths]


pred_file_list = ["./preds/inceptionV3_combine.p", "./preds/vgg16_combine.p"]

pred_list = []
for pred_file in pred_file_list:
    pred = utils.load_pickle(pred_file)
    pred_list.append(pred)
    print(pred.shape)

pred_re = np.sum( np.array(pred_list) ,axis=0)

pred_labels = average_prediction(pred_re, n_crop=n_crop, is_flip=True)

pred_sub = [{"image_id": id, "disease_class": int(label)} for id, label in zip(test_ids, pred_labels)]

utils.dump_to_json("./cv_combine.json", pred_sub)
