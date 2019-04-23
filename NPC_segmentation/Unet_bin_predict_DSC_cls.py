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
PATHB = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/data/patient_2D_predict_dsc_train/train_B/'

SAVEPATH = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/model/train_on_batch_20181229_1_CLS/'
if os.path.exists(SAVEPATH) == 0:
    os.mkdir(SAVEPATH)
CLS_DATA_LIST = os.listdir(PATHB)

clsnum = len(CLS_DATA_LIST)

image_x = 256
image_y = 256

## Model

segmodel = all_model.Unet(input_shape=(image_x,image_y,1))
segmodel.compile(optimizer=Adam(1e-4), loss=all_loss.DiceCoefLoss,
                         metrics=['acc', all_index.precision, all_index.recall, all_index.fmeasure, all_index.yt_sum])
# segmodel.load_weights('/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/model/train_on_batch_20181228_1/seg/SEG_29600.h5')

clsmodel = all_model.Validnet(input_shape=(image_x,image_y,1))
clsmodel.compile(optimizer=SGD(1e-4), loss='categorical_crossentropy')

print(segmodel.summary())
print(clsmodel.summary())

segcount = 0
clscount = 0

iter_num = 0
max_iteration = 10000

bc = 0

batch_size = 4

print('Training start!')
start_time = time.time()
while True:

    fileB = PATHB + CLS_DATA_LIST[clscount%clsnum]

    clscount += 1

    f = h5py.File(fileB)
    # t1_1 = f['t1'][:]
    # t1_1 = np.transpose(t1_1,[1,0,2])
    t2_B = f['t2'][:]
    t2_B = np.transpose(t2_B, [1, 0, 2])
    label_B = f['gt'][:]
    label_B = np.transpose(label_B, [1, 0, 2])
    f.close()

    t2_B = np.expand_dims(t2_B, axis=0)
    label_B = np.expand_dims(label_B, axis=0)

    if bc == 0:
        data_t2_B = copy.deepcopy(t2_B)
        label_train_B = copy.deepcopy(label_B)
    elif bc > 0:
        data_t2_B = np.concatenate([data_t2_B, t2_B], axis=0)
        label_train_B = np.concatenate([label_train_B, label_B], axis=0)

    bc += 1

    if bc < batch_size:
        continue

    bc = 0

    segB_scores = segmodel.predict_on_batch(data_t2_B)
    segB = copy.deepcopy(segB_scores)
    segB[segB>=0.5] = 1
    segB[segB < 0.5] = 0

    dsc_train = np.zeros([batch_size,3],dtype='float')
    for i in range(batch_size):
        segB_one = segB[i,:,:,:]
        label_train_B_one = label_train_B[i,:,:,:]
        segB_p = np.sum(segB_one)
        labelB_p = np.sum(label_train_B_one)
        tp = np.sum(segB_one*label_train_B_one)
        fp = segB_p-tp
        fn = labelB_p - tp
        dsc = 2*tp/(2*tp+fp+fn)
        if dsc>=0.8:
            dsc_train[i, 0] = 1
        elif dsc <= 0.5:
            dsc_train[i, 2] = 1
        else:
            dsc_train[i, 1] = 1
        # dsc_train[i,0] = dsc

    # cls_result = clsmodel.predict_on_batch([data_t2_B, segB_scores])
    cls_result = clsmodel.train_on_batch([data_t2_B,segB_scores],dsc_train)
    print('CLS iteration ', iter_num, '/', max_iteration, ' ', cls_result)
    cls_scores = clsmodel.predict_on_batch([data_t2_B, segB_scores])
    print('True dsc: ', np.transpose(dsc_train, [0, 1]))
    print('Predict dsc: ', np.transpose(cls_scores, [0, 1]))


    iter_num += 1

    if iter_num%400 == 0:

        clsmodel.save_weights(SAVEPATH+'CLS_'+str(iter_num)+'.h5')

    if iter_num>max_iteration:
        end_time = time.time()
        print('Training over, spend ',end_time-start_time)
        break