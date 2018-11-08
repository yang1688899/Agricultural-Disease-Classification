import glob
from keras.models import load_model
from math import ceil
import numpy as np

import config
import utils


batch_size = 1

val_paths = glob.glob(config.VAL_DIR+"*.jpg")
val_ids = [path.split("\\")[-1] for path in val_paths]

num_val = len(val_ids)

model_list = ["./save/inceptionV3_299.model","./save/inceptionResV2_299.model"]
predectiions_list=[]
for model_path in model_list:
    val_test_gen = utils.test_generator(val_paths, batch_size)

    #load model
    model = load_model(model_path)

    predictions = model.predict_generator(val_test_gen,
                            steps=ceil(len(val_ids) / batch_size))
    predectiions_list.append(predictions)


pred_labels = utils.average_prediction_multi(predectiions_list,num_val,n_crop=3,is_flip=True)

pred_list = [{"image_id":id,"disease_class":int(label)} for id,label in zip(val_ids,pred_labels)]

utils.dump_to_json("./val_multi.json",pred_list)

utils.save_to_pickle(predectiions_list,"./predictions_list.p")

# model_1 = load_model("./save/inceptionV3_299.model")
# model_2 = load_model("./save/inceptionResV2_299.model")
#
# # val_gen = utils.val_generator(val_paths,val_ids,batch_size,is_shuffle=False)
#
# for i in range(100):
#     feature = next(val_test_gen)
#     # feature_v,_ = next(val_gen)
#     predict_1 = model_1.predict(feature)
#     predict_2  = model_2.predict(feature)
#
#     # print(type(feature_t[0][0][0][0]))
#     # print(type(feature_v[0][0][0][0]))
#
#     print(np.argmax(predict_1,axis=1))
#     print(np.argmax(predict_2,axis=1))
#     print(np.argmax( (predict_1+predict_2),axis=1 ))
# #
# def average_prediction(predictions,n_crop=3,is_flip=True):
#     if is_flip:
#         num = n_crop*4
#     else:
#         num = n_crop
#     pre_labels = []
#     for i in range(0, len(predictions), num):
#         batch_preds = predictions[i:i + num]
#         sum_pred = np.sum(batch_preds,axis=0)
#         pre_labels.append(np.argmax(sum_pred))
#     return pre_labels
#
# p_list = utils.load_pickle("./predictions_list.p")
# p1 = p_list[0]
# p2= p_list[1]
# for pred_1,pred_2 in zip(p1,p2):
#     print(np.argmax(pred_1))
#     print(np.argmax(pred_2))