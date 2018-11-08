# from sklearn.model_selection import StratifiedKFold
#
# import utils
# import config
#data
# trian_img_paths,train_labels = utils.process_annotation(config.TRAIN_ANNOTATION_FILE,config.TRAIN_DIR)
# val_img_paths,val_labels = utils.process_annotation(config.VAL_ANNOTATION_FILE,config.VAL_DIR)
#
# trian_img_paths.extend(val_img_paths)
# train_labels.extend(val_labels)
#
# #KF_SPLIT
# train_all = []
# evaluate_all = []
# skf = StratifiedKFold(n_splits=5, random_state=1234, shuffle=True)
# # skf.split()
# for train_index, evaluate_index in skf.split(trian_img_paths, train_labels):
#     train_all.append(train_index)
#     evaluate_all.append(evaluate_index)
#     print(train_index.shape,evaluate_index.shape) # the shape is slightly different in different cv, it's OK
#
# #save some object for further use
# save_obj = {"train_all":train_all,"evaluate_all":evaluate_all,"trian_img_paths":trian_img_paths,"train_labels":train_labels}
# print("saving save_obj...")
# utils.save_to_pickle(save_obj,"./save_obj.p")
import numpy as np
import utils
import config

save_obj = utils.load_pickle("./save_obj.p")

trian_img_paths,train_labels = utils.process_annotation(config.TRAIN_ANNOTATION_FILE,config.TRAIN_DIR)
val_img_paths,val_labels = utils.process_annotation(config.VAL_ANNOTATION_FILE,config.VAL_DIR)

trian_img_paths.extend(val_img_paths)
train_labels.extend(val_labels)

def get_cv_data(save_obj,img_paths,labels,num_fold=1):
    train_all = save_obj["train_all"]
    val_all = save_obj["evaluate_all"]
    img_paths = np.array(img_paths)
    labels = np.array(labels)

    train_index = train_all[num_fold-1]
    val_index = val_all[num_fold-1]

    train_paths = img_paths[train_index]
    train_labels = labels[train_index]

    val_paths = img_paths[val_index]
    val_labels = labels[val_index]

    return train_paths,train_labels, val_paths,val_labels

train_paths,train_labels, val_paths,val_labels = get_cv_data(save_obj,trian_img_paths,train_labels,num_fold=1)

print(len(train_paths),len(train_labels))
print(len(val_paths),len(val_labels))