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
PATHA = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/data/patient_2D_predict_dsc_train/train_A/'
PATHB = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/data/patient_2D_predict_dsc_train/train_B/'

SAVEPATH = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/model/train_on_batch_20181228_1/'

SEG_DATA_LIST = os.listdir(PATHA)
CLS_DATA_LIST = os.listdir(PATHB)

segnum = len(SEG_DATA_LIST)
clsnum = len(CLS_DATA_LIST)

image_x = 256
image_y = 256

## Model

segmodel = all_model.Unet(input_shape=(image_x,image_y,1))
segmodel.compile(optimizer=Adam(1e-4), loss=all_loss.DiceCoefLoss,
                         metrics=['acc', all_index.precision, all_index.recall, all_index.fmeasure, all_index.yt_sum])

clsmodel = all_model.Validnet(input_shape=(image_x,image_y,1))
clsmodel.compile(optimizer=Adam(1e-3), loss=all_loss.EuclideanLoss)

print(segmodel.summary())
print(clsmodel.summary())

segcount = 0
clscount = 0

iter_num = 0
max_iteration = 35000

bc = 0

batch_size = 4

print('Training start!')
start_time = time.time()
while True:

    fileA = PATHA + SEG_DATA_LIST[segcount%segnum]
    fileB = PATHB + CLS_DATA_LIST[clscount%clsnum]

    segcount += 1
    clscount += 1

    f = h5py.File(fileA)
    # t1_1 = f['t1'][:]
    # t1_1 = np.transpose(t1_1,[1,0,2])
    t2_A = f['t2'][:]
    t2_A = np.transpose(t2_A, [1, 0, 2])
    label_A = f['gt'][:]
    label_A = np.transpose(label_A, [1, 0, 2])
    f.close()

    f = h5py.File(fileB)
    # t1_1 = f['t1'][:]
    # t1_1 = np.transpose(t1_1,[1,0,2])
    t2_B = f['t2'][:]
    t2_B = np.transpose(t2_B, [1, 0, 2])
    label_B = f['gt'][:]
    label_B = np.transpose(label_B, [1, 0, 2])
    f.close()

    t2_A = np.expand_dims(t2_A, axis=0)
    label_A = np.expand_dims(label_A, axis=0)
    t2_B = np.expand_dims(t2_B, axis=0)
    label_B = np.expand_dims(label_B, axis=0)

    if bc == 0:
        data_t2_A = copy.deepcopy(t2_A)
        label_train_A = copy.deepcopy(label_A)
        data_t2_B = copy.deepcopy(t2_B)
        label_train_B = copy.deepcopy(label_B)
    elif bc > 0:
        data_t2_A = np.concatenate([data_t2_A, t2_A], axis=0)
        label_train_A = np.concatenate([label_train_A, label_A], axis=0)
        data_t2_B = np.concatenate([data_t2_B, t2_B], axis=0)
        label_train_B = np.concatenate([label_train_B, label_B], axis=0)

    bc += 1

    if bc < batch_size:
        continue

    bc = 0

    seg_result = segmodel.train_on_batch(data_t2_A,label_train_A)
    print('SEG iteration ',iter_num,'/',max_iteration,' ',seg_result)

    if iter_num<350:
        iter_num += 1
        continue

    segB_scores = segmodel.predict_on_batch(data_t2_B)
    segB = copy.deepcopy(segB_scores)
    segB[segB>=0.5] = 1
    segB[segB < 0.5] = 0

    dsc_train = np.zeros([batch_size,1],dtype='float')
    for i in range(batch_size):
        segB_one = segB[i,:,:,:]
        label_train_B_one = label_train_B[i,:,:,:]
        segB_p = np.sum(segB_one)
        labelB_p = np.sum(label_train_B_one)
        tp = np.sum(segB_one*label_train_B_one)
        fp = segB_p-tp
        fn = labelB_p - tp
        dsc = 2*tp/(2*tp+fp+fn)
        dsc_train[i,0] = dsc

    print('dsc: ',np.transpose(dsc_train,[1,0]))

    # cls_result = clsmodel.predict_on_batch([data_t2_B, segB_scores])
    cls_result = clsmodel.train_on_batch([data_t2_B,segB_scores],dsc_train)
    print('CLS iteration ', iter_num, '/', max_iteration, ' ', cls_result)

    iter_num += 1

    if iter_num%400 == 0:

        segmodel.save_weights(SAVEPATH+'seg/'+'SEG_'+str(iter_num)+'.h5')
        clsmodel.save_weights(SAVEPATH+'cls/'+'CLS_'+str(iter_num)+'.h5')

    if iter_num>max_iteration:
        end_time = time.time()
        print('Training over, spend ',end_time-start_time)
        break