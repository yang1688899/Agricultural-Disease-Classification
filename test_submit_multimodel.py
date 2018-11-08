import glob
from keras.models import load_model
from math import ceil
import numpy as np

import config
import utils

def average_prediction_multi(predictions_list,num_test,n_crop=3,is_flip=True):
    if is_flip:
        num = n_crop*4
    else:
        num = n_crop
    pre_labels = []
    all_pred_sum = np.zeros(shape=[num_test,61])
    for predictions in predictions_list:
        model_pred_sum = []
        for i in range(0, len(predictions), num):
            batch_preds = predictions[i:i + num]
            sum_pred = np.sum(batch_preds,axis=0)
            model_pred_sum.append(sum_pred)
        model_pred_sum += np.array(model_pred_sum)
    pre_labels = np.argmax(all_pred_sum,axis=1)
    return pre_labels


batch_size = 3

test_paths = glob.glob(config.TEST_DIR+"*.jpg")
test_ids = [path.split("\\")[-1] for path in test_paths]

num_test = len(test_ids)

model_list = []
predectiions_list=[]
for model_path in model_list:
    test_gen = utils.test_generator(test_paths, batch_size)

    #load model
    model = load_model(model_path)

    predictions = model.predict_generator(test_gen,
                            steps=ceil(len(test_ids) / batch_size))
    predectiions_list.append(predictions)


pred_labels = average_prediction_multi(predectiions_list,num_test,n_crop=3,is_flip=True)

pred_list = [{"image_id":id,"disease_class":int(label)} for id,label in zip(test_ids,pred_labels)]

utils.dump_to_json("./submit_interceptionV3.json",pred_list)

utils.save_to_pickle(predictions,"./predictions.p")

# predictions = utils.load_pickle("./predictions.p")
# pred_labels = average_prediction(predictions,n_crop=3,is_flip=True)
# pred_list = [{"image_id":id,"disease_class":int(label)} for id,label in zip(test_ids,pred_labels)]
#
# utils.dump_to_json("./submit.json",pred_list)
#
# print(len(test_ids))
# print(len(predictions)/12)
