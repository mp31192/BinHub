## Bin Huang, made in 2018.8.31

import all_model_bin_3D as all_model
import all_loss_bin as all_loss
import all_index_bin_3D as all_index
from all_callbacks_bin import *
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import SimpleITK as sitk

from keras.optimizers import *
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.utils import multi_gpu_model
from all_optimizer_bin import LRFinder,SGDRScheduler,CyclicLR
from keras.preprocessing.image import ImageDataGenerator

import time
from keras.callbacks import *
import cal_index
import read_data

from read_data_generator_Real3D import *
# Setting the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

image_x = 256
image_y = 256
image_z = 64
gpus_used = 2
batch_size = 1
batch_size_cls = 8

def main_train_real3D():
    train_PathH5 = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/data/patient_3D_train2/'   ##train data , *.h5
    trainset_num = len(os.listdir(train_PathH5))
    evalu_PathH5 = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/data/patient_3D_test2/'##evaluate data , *.h5
    evaluset_num = len(os.listdir(evalu_PathH5))
    model_Pathsave = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/model/fit_seg_20190120_2_3D/'    ##Path to save model
    if os.path.exists(model_Pathsave) == 0:
        os.mkdir(model_Pathsave)
    tensorboard_Path = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/graph/fit_seg_20190120_2_3D'                ##Path to save tensorboard graph
    if os.path.exists(tensorboard_Path) == 0:
        os.mkdir(tensorboard_Path)

    model = all_model.RealThreeDNet_multi_small_single(input_shape=(image_x,image_y,image_z,1))     ##model construction
    print(model.summary())
    # model_multi_gpu = multi_gpu_model(model,gpus=gpus_used)   ##set multi gpu
    ##callbacks
    callback_checkpoint = ModelCheckpoint(filepath = model_Pathsave+"weights.{epoch:02d}-{val_loss:.2f}.hdf5",monitor='val_loss',verbose=0,save_best_only=False,
                                          save_weights_only=False,mode='min',period=3)
    callback_tensorboard = TensorBoard(log_dir=tensorboard_Path,histogram_freq=0,write_graph=True,write_images=True)
    callback_reduceLR = ReduceLROnPlateau(monitor='val_loss',patience=10,mode='auto',factor=0.1,cooldown=5,min_lr=1e-5)
    # callback_cycle_lr = SGDRScheduler(min_lr=1e-5,max_lr=1e-3,steps_per_epoch=np.ceil(trainset_num//batch_size), lr_decay=0.9, cycle_length=3,mult_factor=1.5)
    ##compile model
    model.compile(optimizer=Adam(1e-2), loss=all_loss.DiceCoefLoss,#loss_weights=[1],
                         metrics=['acc', all_index.precision, all_index.recall, all_index.fmeasure, all_index.yt_sum])
    # model.load_weights('/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/model/fit_seg_20181207_1_3D/weights.392-0.45.hdf5')

    ##data generator
    train_generator = generate_segmentation_data_from_file_t1_3D(train_PathH5,batch_size=batch_size)
    valid_generator = generate_segmentation_data_from_file_valid_t1_3D(evalu_PathH5,batch_size=1)
    # image_datagen = ImageDataGenerator(horizontal_flip=True)
    ##train model
    # model.fit_generator(train_generator, callbacks=[callback_checkpoint, callback_tensorboard]
    #                     , steps_per_epoch=trainset_num // batch_size, epochs=100, max_queue_size=1)
    model.fit_generator(train_generator,validation_data=valid_generator
                        ,callbacks=[callback_checkpoint,callback_tensorboard]
                        ,steps_per_epoch=trainset_num//batch_size,initial_epoch = 0,epochs=200,validation_steps=evaluset_num,max_queue_size=1)
    print('\n')
    # model.evaluate_generator(generate_segmentation_data_from_file_valid(evalu_PathH5,batch_size=1),steps=1235,max_queue_size=1)

def main_train_real3D_multimodal():
    train_PathH5 = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/data/patient_3D_train_20190222_2/'   ##train data , *.h5
    trainset_num = len(os.listdir(train_PathH5))
    evalu_PathH5 = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/data/patient_3D_test_20190222_2/'##evaluate data , *.h5
    evaluset_num = len(os.listdir(evalu_PathH5))
    model_Pathsave = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/model/fit_seg_20190329_2_3D/'    ##Path to save model
    if os.path.exists(model_Pathsave) == 0:

        os.mkdir(model_Pathsave)
    tensorboard_Path = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/graph/fit_seg_20190329_2_3D'                ##Path to save tensorboard graph
    if os.path.exists(tensorboard_Path) == 0:
        os.mkdir(tensorboard_Path)

    model = all_model.RealThreeDNet_multi_small_laji_code_decode_dilate_inception_register(input_shape=(image_x,image_y,image_z,1))     ##model construction
    print(model.summary())
    # model_multi_gpu = multi_gpu_model(model,gpus=gpus_used)   ##set multi gpu
    ##callbacks
    callback_checkpoint = ModelCheckpoint(filepath = model_Pathsave+"weights.{epoch:02d}-{val_loss:.2f}.hdf5",monitor='val_loss',verbose=0,save_best_only=False,
                                          save_weights_only=False,mode='min',period=3)
    callback_tensorboard = TensorBoard(log_dir=tensorboard_Path,histogram_freq=0,write_graph=True,write_images=True)
    callback_reduceLR = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', factor=0.1, cooldown=5,
                                          min_lr=1e-4)
    # callback_cycle_lr = SGDRScheduler(min_lr=1e-5,max_lr=1e-3,steps_per_epoch=np.ceil(trainset_num//batch_size), lr_decay=0.9, cycle_length=3,mult_factor=1.5)
    ##compile model
    model.compile(optimizer=Adam(1e-3), loss=[all_loss.DiceCoefLoss],loss_weights=[1],
                         metrics=['acc', all_index.precision, all_index.recall, all_index.fmeasure, all_index.yt_sum])
    model.load_weights('/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/model/fit_seg_20190329_2_3D/weights.300-0.26.hdf5')

    ##data generator
    train_generator = generate_segmentation_data_from_file_multi_3D(train_PathH5,batch_size=batch_size)
    valid_generator = generate_segmentation_data_from_file_valid_multi_3D(evalu_PathH5,batch_size=1)
    # image_datagen = ImageDataGenerator(horizontal_flip=True)
    ##train model
    # model.fit_generator(train_generator, callbacks=[callback_checkpoint, callback_tensorboard]
    #                     , steps_per_epoch=trainset_num // batch_size, epochs=100, max_queue_size=1)
    model.fit_generator(train_generator,validation_data=valid_generator
                        ,callbacks=[callback_checkpoint,callback_tensorboard]
                        ,steps_per_epoch=trainset_num//batch_size,initial_epoch = 300,epochs = 400,validation_steps=evaluset_num,max_queue_size=1)
    print('\n')
    # model.evaluate_generator(generate_segmentation_data_from_file_valid(evalu_PathH5,batch_size=1),steps=1235,max_queue_size=1)

def main_train_real3D_multimodal_bigbatch():
    train_PathH5 = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/data/patient_3D_train2/'   ##train data , *.h5
    trainset_num = len(os.listdir(train_PathH5))
    evalu_PathH5 = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/data/patient_3D_test2/'##evaluate data , *.h5
    evaluset_num = len(os.listdir(evalu_PathH5))
    model_Pathsave = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/model/fit_seg_20190202_1_3D/'    ##Path to save model
    if os.path.exists(model_Pathsave) == 0:
        os.mkdir(model_Pathsave)
    tensorboard_Path = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/graph/fit_seg_20190202_1_3D'                ##Path to save tensorboard graph
    if os.path.exists(tensorboard_Path) == 0:
        os.mkdir(tensorboard_Path)

    model = all_model.RealThreeDNet_multi_small_laji_code_decode_capsules(input_shape=(image_x,image_y,image_z,1))     ##model construction
    print(model.summary())
    multi_model = multi_gpu_model(model,gpus=gpus_used)   ##set multi gpu
    ##callbacks
    # callback_checkpoint = CustomModelCheckpoint(model,model_Pathsave)
    callback_checkpoint = ModelCheckpoint(filepath = model_Pathsave+"weights.{epoch:02d}-{val_loss:.2f}.hdf5",monitor='val_loss',verbose=0,save_best_only=False,
                                          save_weights_only=True,mode='min',period=1)
    callback_tensorboard = TensorBoard(log_dir=tensorboard_Path,histogram_freq=0,write_graph=True,write_images=True)
    callback_reduceLR = ReduceLROnPlateau(monitor='val_loss', patience=20, mode='auto', factor=0.1, cooldown=5,
                                          min_lr=1e-4)
    # callback_cycle_lr = SGDRScheduler(min_lr=1e-5,max_lr=1e-3,steps_per_epoch=np.ceil(trainset_num//batch_size), lr_decay=0.9, cycle_length=3,mult_factor=1.5)
    ##compile model
    multi_model.compile(optimizer=Adam(1e-3), loss=all_loss.DiceCoefLoss,#loss_weights=[1],
                         metrics=['acc', all_index.precision, all_index.recall, all_index.fmeasure, all_index.yt_sum])
    # model.load_weights('/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/model/fit_seg_20190111_1_3D/weights.93-0.58.hdf5')

    ##data generator
    train_generator = generate_segmentation_data_from_file_multi_3D(train_PathH5,batch_size=2)
    valid_generator = generate_segmentation_data_from_file_valid_multi_3D(evalu_PathH5,batch_size=2)
    # image_datagen = ImageDataGenerator(horizontal_flip=True)
    ##train model
    # model.fit_generator(train_generator, callbacks=[callback_checkpoint, callback_tensorboard]
    #                     , steps_per_epoch=trainset_num // batch_size, epochs=100, max_queue_size=1)
    multi_model.fit_generator(train_generator,validation_data=valid_generator
                        ,callbacks=[callback_checkpoint,callback_tensorboard,callback_reduceLR]
                        ,steps_per_epoch=trainset_num//2,initial_epoch = 0,epochs=250,validation_steps=evaluset_num,max_queue_size=2)
    # model.save(model_Pathsave+'model_best.h5')
    print('\n') #trainset_num//batch_size
    # model.evaluate_generator(generate_segmentation_data_from_file_valid(evalu_PathH5,batch_size=1),steps=1235,max_queue_size=1)


def main_train_real3D_multimodal_multiout():
    train_PathH5 = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/data/patient_3D_train2/'   ##train data , *.h5
    trainset_num = len(os.listdir(train_PathH5))
    evalu_PathH5 = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/data/patient_3D_test2/'##evaluate data , *.h5
    evaluset_num = len(os.listdir(evalu_PathH5))
    model_Pathsave = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/model/fit_seg_20190207_2_3D/'    ##Path to save model
    if os.path.exists(model_Pathsave) == 0:
        os.mkdir(model_Pathsave)
    tensorboard_Path = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/graph/fit_seg_20190207_2_3D'                ##Path to save tensorboard graph
    if os.path.exists(tensorboard_Path) == 0:
        os.mkdir(tensorboard_Path)

    model = all_model.RealThreeDNet_multi_small_laji_code_decode_multiout(input_shape=(image_x,image_y,image_z,1))     ##model construction
    print(model.summary())
    # model_multi_gpu = multi_gpu_model(model,gpus=gpus_used)   ##set multi gpu
    ##callbacks
    callback_checkpoint = ModelCheckpoint(filepath = model_Pathsave+"weights.{epoch:02d}-{val_loss:.2f}.hdf5",monitor='val_loss',verbose=0,save_best_only=False,
                                          save_weights_only=False,mode='min',period=3)
    callback_tensorboard = TensorBoard(log_dir=tensorboard_Path,histogram_freq=0,write_graph=True,write_images=True)
    callback_reduceLR = ReduceLROnPlateau(monitor='val_loss', patience=20, mode='auto', factor=0.1, cooldown=5,
                                          min_lr=1e-4)
    # callback_cycle_lr = SGDRScheduler(min_lr=1e-5,max_lr=1e-3,steps_per_epoch=np.ceil(trainset_num//batch_size), lr_decay=0.9, cycle_length=3,mult_factor=1.5)
    ##compile model
    model.compile(optimizer=Adam(1e-3), loss=all_loss.DiceCoefLoss,#loss_weights=[1],
                         metrics=['acc', all_index.precision, all_index.recall, all_index.fmeasure, all_index.yt_sum])
    # model.load_weights('/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/model/fit_seg_20190202_2_3D/weights.45-0.82.hdf5')

    ##data generator
    train_generator = generate_segmentation_data_from_file_multi_3D_multiout(train_PathH5,batch_size=batch_size)
    valid_generator = generate_segmentation_data_from_file_valid_multi_3D_multiout(evalu_PathH5,batch_size=1)
    # image_datagen = ImageDataGenerator(horizontal_flip=True)
    ##train model
    # model.fit_generator(train_generator, callbacks=[callback_checkpoint, callback_tensorboard]
    #                     , steps_per_epoch=trainset_num // batch_size, epochs=100, max_queue_size=1)
    model.fit_generator(train_generator,validation_data=valid_generator
                        ,callbacks=[callback_checkpoint,callback_tensorboard,callback_reduceLR]
                        ,steps_per_epoch=trainset_num//batch_size,initial_epoch = 0,epochs=250,validation_steps=evaluset_num,max_queue_size=1)
    print('\n')
    # model.evaluate_generator(generate_segmentation_data_from_file_valid(evalu_PathH5,batch_size=1),steps=1235,max_queue_size=1)

def main_test():
    model_PathH5 = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/model/fit_seg_20190105_3_3D/weights.168-0.45.hdf5' ##model weight path
    model = load_model(model_PathH5,compile=False)  ##load model construction
    # image_savePath = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/image_show/20181008_1/'
    # if os.path.exists(image_savePath) == 0:
    #     os.mkdir(image_savePath)
    test_PathH5 = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/data/patient_3D_train/'   ##path of saving test data, *.h5
    # test_PathH5 = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/data/patient_test_person_3D/66/'  ##path of saving test data, *.h5
    test_num = len(os.listdir(test_PathH5))

    ##compile model
    model.compile(optimizer=Adam(lr=1.0e-5), loss='binary_crossentropy',
                         metrics=['acc', all_index.precision, all_index.recall, all_index.fmeasure, all_index.yt_sum,
                                  all_index.tp,all_index.tn,all_index.fp,all_index.fn])

    ##test model
    result = model.evaluate_generator(generate_segmentation_data_from_file_valid_multi_3D(test_PathH5,batch_size=1),steps=test_num//1,verbose=1)
    print(result)

    # ##test model on batch
    # tp = 0
    # tn = 0
    # fp = 0
    # fn = 0

    ## cal recall,precision,dsc
    recall,precision,dsc = cal_index.cal_seg_index(result,test_num)
    print('recall:',recall,' precision:',precision,' dsc:',dsc,'\n')

def main_test_multimodal():
    # model = all_model.RealThreeDNet_multi_small_laji_code_decode(input_shape=(image_x,image_y,image_z,1))     ##model construction
    model_PathH5 = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/model/fit_seg_20190315_2_3D/weights.237-0.68.hdf5' ##model weight path
    # model = multi_gpu_model(model, gpus=gpus_used)  ##set multi gpu
    # model.load_weights(model_PathH5,by_name=False)
    model = load_model(model_PathH5,compile=False)  ##load model construction
    # image_savePath = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/image_show/20181008_1/'
    # if os.path.exists(image_savePath) == 0:
    #     os.mkdir(image_savePath)
    test_PathH5 = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/data/patient_3D_test_20190222_2/'   ##path of saving test data, *.h5
    # test_PathH5 = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/data/patient_test_person_3D/66/'  ##path of saving test data, *.h5
    test_num = len(os.listdir(test_PathH5))

    ##compile model
    model.compile(optimizer=Adam(lr=1.0e-5), loss='binary_crossentropy',
                         metrics=[all_index.tp,all_index.tn,all_index.fp,all_index.fn,all_index.precision, all_index.recall, all_index.fmeasure])

    ##test model
    result = model.evaluate_generator(generate_segmentation_data_from_file_valid_multi_3D(test_PathH5,batch_size=1),steps=test_num//1,verbose=1)
    print(result)

    # ##test model on batch
    # tp = 0
    # tn = 0
    # fp = 0
    # fn = 0

    ## cal recall,precision,dsc
    recall,precision,dsc = cal_index.cal_seg_index(result[0:5],test_num)
    print('recall:',recall,' precision:',precision,' dsc:',dsc,'\n')

def main_test_multimodal_multiout():
    model_PathH5 = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/model/fit_seg_20190207_2_3D/weights.120-0.72.hdf5' ##model weight path
    model = load_model(model_PathH5,compile=False)  ##load model construction
    # image_savePath = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/image_show/20181008_1/'
    # if os.path.exists(image_savePath) == 0:
    #     os.mkdir(image_savePath)
    test_PathH5 = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/data/patient_3D_test3/'   ##path of saving test data, *.h5
    # test_PathH5 = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/data/patient_test_person_3D/66/'  ##path of saving test data, *.h5
    test_num = len(os.listdir(test_PathH5))

    ##compile model
    model.compile(optimizer=Adam(lr=1.0e-5), loss='binary_crossentropy',
                         metrics=[all_index.tp,all_index.tn,all_index.fp,all_index.fn])

    ##test model
    result = model.evaluate_generator(generate_segmentation_data_from_file_valid_multi_3D_multiout(test_PathH5,batch_size=1),steps=test_num//1,verbose=1)
    print(result)

    # ##test model on batch
    # tp = 0
    # tn = 0
    # fp = 0
    # fn = 0

    ## cal recall,precision,dsc
    recall,precision,dsc = cal_index.cal_seg_index(result[2:7],test_num)
    print('recall:',recall,' precision:',precision,' dsc:',dsc,'\n')

def main_test_bypredict():
    seg_thresh = 0.5
    model_PathH5 = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/model/fit_seg_20190105_3_3D/weights.168-0.45.hdf5' ##model weight path
    model = load_model(model_PathH5,compile=False)  ##load model construction

    image_savePath = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/image_show/20190105_3_3D_multi/'
    if os.path.exists(image_savePath) == 0:
        os.mkdir(image_savePath)

    test_PathH5 = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/data/patient_3D_test/'   ##path of saving test data, *.h5
    testList = os.listdir(test_PathH5)
    test_num = len(testList)

    ##compile model
    # model.compile(optimizer=Adam(lr=1.0e-5), loss='binary_crossentropy',
    #                      metrics=['acc', all_index.precision, all_index.recall, all_index.fmeasure, all_index.yt_sum,
    #                               all_index.tp,all_index.tn,all_index.fp,all_index.fn])

    # ##test model predict on batch
    tpall = np.zeros([1,],dtype='float')
    tnall = np.zeros([1,],dtype='float')
    fpall = np.zeros([1,],dtype='float')
    fnall = np.zeros([1,],dtype='float')
    target_num = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for num in range(test_num):

        FullPath = test_PathH5+testList[num]
        data_list = read_data.read_data_h5(FullPath, 't2', 't1', 'gt')
        data_t2 = data_list[0]
        data_t2 = data_t2*1
        data_t1 = data_list[1]
        data_t1 = data_t1*1
        data_gt = data_list[2]

        if np.sum(data_gt) == 0:
            continue

        data_t2 = np.transpose(data_t2,[1,0,2])
        data_t1 = np.transpose(data_t1, [1, 0, 2])
        data_gt = np.transpose(data_gt, [1, 0, 2])

        data_t2 = np.expand_dims(data_t2,axis=0)
        data_t1 = np.expand_dims(data_t1, axis=0)
        data_gt = np.expand_dims(data_gt, axis=0)
        result_ori = model.predict_on_batch([data_t2])
        result = copy.deepcopy(result_ori)
        seg_result = copy.deepcopy(result)
        seg_result[seg_result>=seg_thresh] = 1
        seg_result[seg_result<seg_thresh] = 0

        r, c = 2, 2
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                if i == 0:
                    if j == 0:
                        axs[i, j].imshow(data_t2[cnt, :, :, 0], cmap='gray')
                        axs[i, j].axis('off')
                    elif j == 1:
                        axs[i, j].imshow(data_gt[cnt, :, :, 0], cmap='gray')
                        axs[i, j].axis('off')
                elif i == 1:
                    if j == 0:
                        axs[i, j].imshow(result[cnt, :, :, 0], cmap='gray')
                        axs[i, j].axis('off')
                    elif j == 1:
                        axs[i, j].imshow(seg_result[cnt, :, :, 0], cmap='gray')
                        axs[i, j].axis('off')

        label_target = copy.deepcopy(data_gt[0,:,:,target_num])
        seg_target = copy.deepcopy(seg_result[0,:,:,target_num])
        true_point = np.sum(label_target)
        pred_point = np.sum(seg_target)

        tp = np.sum(seg_target*label_target)
        fp = pred_point-tp
        fn = true_point-tp
        tn = image_x*image_y-tp-fp-fn

        tpall[target_num] = tp+tpall[target_num]
        tnall[target_num] = tn+tnall[target_num]
        fpall[target_num] = fp+fpall[target_num]
        fnall[target_num] = fn+fnall[target_num]
        ## cal recall,precision,dsc
        recall,precision,dsc = cal_index.cal_rpd(tp,tn,fp,fn)
        print(target_num,' recall:',recall,' precision:',precision,' dsc:',dsc,' true_all:',true_point,' pred_all:',pred_point)

        # if dsc<0.5:
        #     fig.savefig(image_savePath_bad + testList[num] + '_' + str(dsc) +'_.png')
        #     plt.close()
        # else:
        #     fig.savefig(image_savePath + testList[num] + '_' + str(dsc) + '_.png')
        #     plt.close()

        ## cal all recall,precision,dsc
    recall, precision, dsc = cal_index.cal_rpd(tpall[target_num], tnall[target_num], fpall[target_num], fnall[target_num])
    print(target_num,' ALL recall:', recall, ' precision:', precision, ' dsc:', dsc)

def main_test_predict_save():
    seg_thresh = 0.5
    model_PathH5 = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/model/fit_seg_20190107_3_3D/weights.168-0.97.hdf5'  ##model weight path
    model = load_model(model_PathH5, compile=False)  ##load model construction

    image_savePath = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/image_show/20190107_3_3D_multi/train/'
    if os.path.exists(image_savePath) == 0:
        os.mkdir(image_savePath)

    test_PathH5 = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/data/patient_3D_train2/'  ##path of saving test data, *.h5
    testList = os.listdir(test_PathH5)
    test_num = len(testList)

    ##compile model
    # model.compile(optimizer=Adam(lr=1.0e-5), loss='binary_crossentropy',
    #                      metrics=['acc', all_index.precision, all_index.recall, all_index.fmeasure, all_index.yt_sum,
    #                               all_index.tp,all_index.tn,all_index.fp,all_index.fn])

    # ##test model predict on batch

    target_num = 0

    for num in range(test_num):

        Filename = testList[num]


        FullPath = test_PathH5 + testList[num]
        data_list = read_data.read_data_h5(FullPath, 't2', 'rt1', 'gt')
        data_t2 = data_list[0]
        data_t2 = data_t2 * 1
        data_t1 = data_list[1]
        data_t1 = data_t1 * 1
        data_gt = data_list[2]

        max_t1 = np.max(data_t1)
        min_t1 = np.min(data_t1)
        max_t2 = np.max(data_t2)
        min_t2 = np.min(data_t2)

        data_t1 = (data_t1-min_t1)/(max_t1-min_t1)
        data_t2 = (data_t2 - min_t2) / (max_t2 - min_t2)

        label_shape = np.shape(data_gt)

        t1_1 = np.zeros([256, 256, 64], dtype='float')
        t2_1 = np.zeros([256,256,64],dtype='float')
        label_1 = np.zeros([256, 256, 64], dtype='float')

        t1_1[:, :, 4:4 + label_shape[2]] = data_t1[:, :, :, 0]
        t2_1[:,:,4:4+label_shape[2]] = data_t2[:,:,:,0]
        label_1[:, :, 4:4+label_shape[2]] = data_gt[:, :, :,0]

        t1_1 = np.expand_dims(t1_1, axis=4)
        t2_1 = np.expand_dims(t2_1, axis=4)
        label_1 = np.expand_dims(label_1, axis=4)

        t1_1 = np.expand_dims(t1_1, axis=0)
        t2_1 = np.expand_dims(t2_1, axis=0)
        label_1 = np.expand_dims(label_1, axis=0)


        result_ori = model.predict_on_batch([t2_1,t1_1])
        result = copy.deepcopy(result_ori)
        seg_result = copy.deepcopy(result)
        seg_result[seg_result >= seg_thresh] = 1
        seg_result[seg_result < seg_thresh] = 0

        label_target = copy.deepcopy(label_1[0, :, :, :, 0])
        seg_target = copy.deepcopy(seg_result[0, :, :, :, 0])

        true_point = np.sum(label_target)
        pred_point = np.sum(seg_target)

        tp = np.sum(seg_target * label_target)
        fp = pred_point - tp
        fn = true_point - tp
        tn = image_x * image_y - tp - fp - fn

        ## cal recall,precision,dsc
        recall, precision, dsc = cal_index.cal_rpd(tp, tn, fp, fn)
        print(target_num, ' recall:', recall, ' precision:', precision, ' dsc:', dsc, ' true_all:', true_point,
              ' pred_all:', pred_point)

        image_savePath_patient = image_savePath+Filename[:-3]+'_'+str(dsc)
        if os.path.exists(image_savePath_patient) == 0:
            os.mkdir(image_savePath_patient)

        for z in range(64):
            r, c = 2, 2
            fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    if i == 0:
                        if j == 0:
                            axs[i, j].imshow(t2_1[cnt, :, :, z, 0], cmap='gray')
                            axs[i, j].axis('off')
                        elif j == 1:
                            axs[i, j].imshow(t1_1[cnt, :, :, z, 0], cmap='gray')
                            axs[i, j].axis('off')
                    elif i == 1:
                        if j == 0:
                            axs[i, j].imshow(label_1[cnt, :, :, z, 0], cmap='gray')
                            axs[i, j].axis('off')
                        elif j == 1:
                            axs[i, j].imshow(result_ori[cnt, :, :, z,0], cmap='gray')
                            axs[i, j].axis('off')
            fig.savefig(image_savePath_patient + '/' + str(z) + '.png')
            plt.close()

def main_test_predict_save_nii():
    seg_thresh = 0.5
    model_PathH5 = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/model/fit_seg_20190315_1_3D/weights.183-0.33.hdf5'  ##model weight path
    model = load_model(model_PathH5, compile=False)  ##load model construction

    image_savePath = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/image_show/20190315_1_3D/test/'
    if os.path.exists(image_savePath) == 0:
        os.makedirs(image_savePath)

    test_PathH5 = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/data/patient_3D_test_20190222_2/'  ##path of saving test data, *.h5
    testList = os.listdir(test_PathH5)
    test_num = len(testList)

    ##compile model
    # model.compile(optimizer=Adam(lr=1.0e-5), loss='binary_crossentropy',
    #                      metrics=['acc', all_index.precision, all_index.recall, all_index.fmeasure, all_index.yt_sum,
    #                               all_index.tp,all_index.tn,all_index.fp,all_index.fn])

    # ##test model predict on batch

    target_num = 0

    for num in range(test_num):

        ## get patient number
        Filename = testList[num]
        Filename = Filename.split('.')
        Filename = Filename[0]

        FullPath = test_PathH5 + testList[num]
        data_list = read_data.read_data_h5(FullPath, 't2', 't1', 'gt')
        data_t2 = data_list[0]
        data_t2 = data_t2 * 1
        data_t2 = np.transpose(data_t2, [1, 0, 2, 3])
        data_t1 = data_list[1]
        data_t1 = data_t1 * 1
        data_t1 = np.transpose(data_t1, [1, 0, 2, 3])
        data_gt = data_list[2]
        data_gt = np.transpose(data_gt, [1, 0, 2, 3])

        max_t1 = np.max(data_t1)
        min_t1 = np.min(data_t1)
        max_t2 = np.max(data_t2)
        min_t2 = np.min(data_t2)

        data_t1 = (data_t1-min_t1)/(max_t1-min_t1)
        data_t2 = (data_t2 - min_t2) / (max_t2 - min_t2)

        label_shape = np.shape(data_gt)

        t1_1 = np.zeros([256, 256, 64], dtype='float')
        t2_1 = np.zeros([256,256,64],dtype='float')
        label_1 = np.zeros([256, 256, 64], dtype='float')

        t1_1[:, :, 4:4 + label_shape[2]] = data_t1[:, :, :, 0]
        t2_1[:,:,4:4+label_shape[2]] = data_t2[:,:,:,0]
        label_1[:, :, 4:4+label_shape[2]] = data_gt[:, :, :,0]

        t1_1 = np.expand_dims(t1_1, axis=4)
        t2_1 = np.expand_dims(t2_1, axis=4)20
        label_1 = np.expand_dims(label_1, axis=4)

        t1_1 = np.expand_dims(t1_1, axis=0)
        t2_1 = np.expand_dims(t2_1, axis=0)
        label_1 = np.expand_dims(label_1, axis=0)

        result_ori = model.predict_on_batch([t2_1,t1_1])
        result = copy.deepcopy(result_ori)
        seg_result = copy.deepcopy(result)
        seg_result[seg_result >= seg_thresh] = 1
        seg_result[seg_result < seg_thresh] = 0

        label_target = copy.deepcopy(label_1[0, :, :, :, 0])
        seg_target = copy.deepcopy(seg_result[0, :, :, :, 0])

        true_point = np.sum(label_target)
        pred_point = np.sum(seg_target)

        tp = np.sum(seg_target * label_target)
        fp = pred_point - tp
        fn = true_point - tp
        tn = image_x * image_y - tp - fp - fn

        ## cal recall,precision,dsc
        recall, precision, dsc = cal_index.cal_rpd(tp, tn, fp, fn)
        print(target_num, ' recall:', recall, ' precision:', precision, ' dsc:', dsc, ' true_all:', true_point,
              ' pred_all:', pred_point)
        dsc_str = str(dsc)
        image_savePath_patient = image_savePath+Filename+'_'+dsc_str[:4]+'_t1.nii'
        t1_1 = np.transpose(t1_1, [0,3,2,1,4])
        saveArray2nii(t1_1[0,:,:,:,0],image_savePath_patient)
        image_savePath_patient = image_savePath + Filename + '_' + dsc_str[:4] + '_t2.nii'
        t2_1 = np.transpose(t2_1, [0, 3, 2, 1, 4])
        saveArray2nii(t2_1[0, :, :, :, 0], image_savePath_patient)
        image_savePath_patient = image_savePath + Filename + '_' + dsc_str[:4] + '_result.nii'
        result_ori = np.transpose(result_ori, [0, 3, 2, 1, 4])
        saveArray2nii(result_ori[0, :, :, :, 0], image_savePath_patient)
        image_savePath_patient = image_savePath + Filename + '_' + dsc_str[:4] + '_gt.nii'
        label_1 = np.transpose(label_1, [0, 3, 2, 1, 4])
        saveArray2nii(label_1[0, :, :, :, 0], image_savePath_patient)
        # print('end')

def saveArray2nii(image,path):
    image_nii = sitk.GetImageFromArray(image)
    sitk.WriteImage(image_nii, path)

if __name__ == "__main__":
    main_train_real3D_multimodal()


