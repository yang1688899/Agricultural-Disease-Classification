import glob
from keras.models import load_model
from math import ceil
import numpy as np

import config
import utils

def average_prediction(predictions,n_crop=3,is_flip=True):
    if is_flip:
        num = n_crop*2
    else:
        num = n_crop
    pre_labels = []
    for i in range(0, len(test_ids), num):
        batch_preds = predictions[i:i + num]

        sum_pred = np.zeros([num, 61])
        for j in range(num):
            sum_pred += batch_preds[j]
        avg_pred = sum_pred/num
        pre_labels.append(np.argmax(avg_pred))
    return pre_labels

#load model
model = load_model(config.SAVE_MODEL_PATH)

batch_size = 4

test_paths = glob.glob(config.TEST_DIR+"*.jpg")
test_ids = [path.split("\\")[-1] for path in test_paths]
test_gen = utils.test_generator(test_paths,batch_size)

predictions = model.predict_generator(test_gen,
                        steps=ceil(len(test_ids) / batch_size))

print(len(predictions))

pred_labels = average_prediction(predictions,n_crop=3,is_flip=True)

pred_list = [{"image_id":id,"disease_class":int(label)} for id,label in zip(test_ids,pred_labels)]

utils.dump_to_json("./submit.json",pred_list)



