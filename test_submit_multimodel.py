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

test_paths = glob.glob(config.TEST_DIR_B+"*.jpg")
test_ids = [path.split("\\")[-1] for path in test_paths]

print(test_ids)
print(len(test_ids))

model_list = ["./save/dense121_224_randangle.model",
              "./save/inceptionV3_299.model",
              "./save/inceptionResV2_299_full_re1.model"]

predictions_list = []
for path in model_list:

    test_gen = utils.test_generator(test_paths, n_crop, batch_size)

    # load model
    model = load_model(path)

    predictions = model.predict_generator(test_gen,
                                          steps=ceil(len(test_ids) / batch_size))
    predictions_list.append(predictions)

predictions_combine = np.sum(np.array(predictions_list), axis=0)

utils.save_to_pickle(predictions_combine, "./preds/299.p")
print(len(predictions_combine) / len(test_ids))
