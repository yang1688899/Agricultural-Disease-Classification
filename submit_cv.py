import glob
from keras.models import load_model
from math import ceil
import numpy as np


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

test_paths = glob.glob(config.TEST_DIR+"*.jpg")
test_ids = [path.split("\\")[-1] for path in test_paths]

predictions_list = []
for i in range(folds):
    LOAD_MODEL_PATH = "./save/inceptionV3_299_fold%s.model" % (i + 1)

    test_gen = utils.test_generator(test_paths, n_crop, batch_size)

    # load model
    model = load_model(LOAD_MODEL_PATH)

    predictions = model.predict_generator(test_gen,
                                          steps=ceil(len(test_ids) / batch_size))
    predictions_list.append(predictions)

predictions_combine = np.sum(np.array(predictions_list), axis=0)

utils.save_to_pickle(predictions_combine, "./preds/inceptionV3_combine.p")
print(len(predictions_combine) / len(test_ids))

pred_labels = average_prediction(predictions_combine, n_crop=n_crop, is_flip=True)

pred_list = [{"image_id": id, "disease_class": int(label)} for id, label in zip(test_ids, pred_labels)]

utils.dump_to_json("./val_inceptionV3_299_combine.json", pred_list)