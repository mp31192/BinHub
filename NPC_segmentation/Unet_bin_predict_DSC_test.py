import h5py
import os
import numpy as np
import copy
import time

import all_model_bin as all_model
import all_loss_bin as all_loss
import all_index_bin as all_index

from keras.optimizers import *

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

## Read data
PATHA = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/data/patient_2D_predict_dsc_test/test/'

SEG_DATA_LIST = os.listdir(PATHA)
CLS_DATA_LIST = os.listdir(PATHA)

segnum = len(SEG_DATA_LIST)
clsnum = len(CLS_DATA_LIST)

image_x = 256
image_y = 256

## Model

segmodel = all_model.Unet(input_shape=(image_x,image_y,1))
segmodel.compile(optimizer=Adam(1e-4), loss=all_loss.DiceCoefLoss,
                         metrics=['acc', all_index.precision, all_index.recall, all_index.fmeasure, all_index.yt_sum,
                         all_index.tp,all_index.fp,all_index.tn,all_index.fn])
segmodel.load_weights('/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/model/train_on_batch_20181228_1/seg/SEG_29600.h5')

clsmodel = all_model.Validnet(input_shape=(image_x,image_y,1))
clsmodel.compile(optimizer=Adam(1e-3), loss=all_loss.EuclideanLoss, metrics=['acc'])
clsmodel.load_weights('/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/model/train_on_batch_20181229_1_CLS/CLS_5200.h5')

print(clsmodel.summary())
print(segmodel.summary())

segcount = 0

iter_num = 0

bc = 0

batch_size = 1

print('Test start!')
start_time = time.time()

tp_all = 0
fp_all = 0
fn_all = 0
tn_all = 0

tp_cls_all = 0
fp_cls_all = 0

eu_all = 0

good_sample = 0
bad_sample = 0
mid_sample = 0

for i in range(segnum):

    fileA = PATHA + SEG_DATA_LIST[segcount%segnum]

    segcount += 1

    f = h5py.File(fileA)
    # t1_1 = f['t1'][:]
    # t1_1 = np.transpose(t1_1,[1,0,2])
    t2_A = f['t2'][:]
    t2_A = np.transpose(t2_A, [1, 0, 2])
    label_A = f['gt'][:]
    label_A = np.transpose(label_A, [1, 0, 2])
    f.close()

    t2_A = np.expand_dims(t2_A, axis=0)
    label_A = np.expand_dims(label_A, axis=0)

    if bc == 0:
        data_t2_A = copy.deepcopy(t2_A)
        label_train_A = copy.deepcopy(label_A)
    elif bc > 0:
        data_t2_A = np.concatenate([data_t2_A, t2_A], axis=0)
        label_train_A = np.concatenate([label_train_A, label_A], axis=0)

    bc += 1

    if bc < batch_size:
        continue

    bc = 0

    seg_result = segmodel.test_on_batch(data_t2_A,label_train_A)
    print('SEG TEST ', i, '/', segnum, ' ', seg_result)

    tp_all = tp_all+seg_result[6]
    fp_all = fp_all+seg_result[7]
    tn_all = tn_all+seg_result[8]
    fn_all = fn_all+seg_result[9]

    seg_scores = segmodel.predict_on_batch(data_t2_A)
    cls_result = clsmodel.predict_on_batch([data_t2_A,seg_scores])

    dsc = seg_result[4]
    if dsc >= 0.8:
        dsc_result = 0
        good_sample += 1
    elif dsc <= 0.6:
        dsc_result = 1
        bad_sample += 1
    else:
        dsc_result = 2
        mid_sample += 1

    dsc_predict = np.argmax(cls_result)
    print("True DSC:",dsc," True class:",dsc_result," Predict class:",dsc_predict)
    if dsc_predict==dsc_result:
        tp_cls_all = tp_cls_all + 1
    else:
        fp_cls_all = fp_cls_all + 1

    # eu_all = eu_all+abs(seg_result[4] - cls_result)
    # print('True Dsc:',seg_result[4],' Predict Dsc:',cls_result," Minu:",abs(seg_result[4] - cls_result))


Recall = tp_all/(tp_all+fn_all)
Precision = tp_all/(tp_all+fp_all)
DSC = 2*tp_all/(2*tp_all+fp_all+fn_all+1)

print("DSC:",DSC," Recall:",Recall," Precision:",Precision)
print("CLS ACC:",tp_cls_all/segnum)
print("Good sample:",good_sample," Bad sample:",bad_sample," Mid sample:",mid_sample)
print("Ratio Good sample:",good_sample/segnum," Bad sample:",bad_sample/segnum," Mid sample:",mid_sample/segnum)
# print("Mean Euclidean Distance:",eu_all/segnum)
end_time = time.time()
print("Spend Time:",end_time-start_time)