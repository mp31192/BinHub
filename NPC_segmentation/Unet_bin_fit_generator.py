## Bin Huang, made in 2018.8.31

import all_model_bin as all_model
import all_loss_bin as all_loss
import all_index_bin as all_index
import numpy as np
import matplotlib.pyplot as plt

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

from read_data_generator import *
# Setting the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

image_x = 256
image_y = 256
gpus_used = 2
batch_size = 4
batch_size_cls = 8
def main_train():
    train_PathH5 = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/data/patient_2D_train/'   ##train data , *.h5
    trainset_num = len(os.listdir(train_PathH5))
    evalu_PathH5 = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/data/patient_2D_test/'##evaluate data , *.h5
    evaluset_num = len(os.listdir(evalu_PathH5))
    model_Pathsave = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/model/fit_seg_TEST/'    ##Path to save model
    if os.path.exists(model_Pathsave) == 0:
        os.mkdir(model_Pathsave)
    tensorboard_Path = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/graph/fit_seg_TEST'                ##Path to save tensorboard graph
    if os.path.exists(tensorboard_Path) == 0:
        os.mkdir(tensorboard_Path)

    model = all_model.threeDNet_single_deep(input_shape=(image_x,image_y,1))     ##model construction
    print(model.summary())
    # model_multi_gpu = multi_gpu_model(model,gpus=gpus_used)   ##set multi gpu
    ##callbacks
    callback_checkpoint = ModelCheckpoint(filepath = model_Pathsave+"weights.{epoch:02d}-{val_loss:.2f}.hdf5",monitor='val_loss',verbose=0,save_best_only=False,
                                          save_weights_only=False,mode='min',period=1)
    callback_tensorboard = TensorBoard(log_dir=tensorboard_Path,histogram_freq=0,write_graph=True,write_images=True)
    # callback_cycle_lr = SGDRScheduler(min_lr=1e-5,max_lr=1e-3,steps_per_epoch=np.ceil(trainset_num//batch_size), lr_decay=0.9, cycle_length=3,mult_factor=1.5)
    ##compile model
    model.compile(optimizer=Adam(1e-3), loss=all_loss.DiceCoefLoss,#loss_weights=[1],
                         metrics=['acc', all_index.precision, all_index.recall, all_index.fmeasure, all_index.yt_sum])
    # model.load_weights('/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/model/fit_seg_20181206_2/weights.79-0.09.hdf5')

    ##data generator
    train_generator = generate_segmentation_data_from_file_t2_256(train_PathH5,batch_size=batch_size)
    valid_generator = generate_segmentation_data_from_file_valid_t2(evalu_PathH5,batch_size=1)
    # image_datagen = ImageDataGenerator(horizontal_flip=True)
    ##train model
    # model.fit_generator(train_generator, callbacks=[callback_checkpoint, callback_tensorboard]
    #                     , steps_per_epoch=trainset_num // batch_size, epochs=100, max_queue_size=1)
    model.fit_generator(train_generator,validation_data=valid_generator
                        ,callbacks=[callback_checkpoint,callback_tensorboard]
                        ,steps_per_epoch=trainset_num//batch_size,initial_epoch = 0,epochs=200,validation_steps=evaluset_num/2,max_queue_size=1)
    print('\n')
    # model.evaluate_generator(generate_segmentation_data_from_file_valid(evalu_PathH5,batch_size=1),steps=1235,max_queue_size=1)

def main_train_cls():

    train_PathH5 = '/media/root/01456da2-1f1d-4b67-810e-b9cd3341133d/AI challenge/2018_seg/Edema_trainingset/'   ##train data , *.h5
    trainset_num = len(os.listdir(train_PathH5+'original_images/'))*128
    evalu_PathH5 = '/media/root/01456da2-1f1d-4b67-810e-b9cd3341133d/AI challenge/2018_seg/Edema_validationset/'##evaluate data , *.h5
    evaluset_num = len(os.listdir(evalu_PathH5+'original_images/')) * 128
    model_Pathsave = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/2018_AI_challenge_seg/model/fit_cls_only_20180912_1/'    ##Path to save model
    if os.path.exists(model_Pathsave) == 0:
        os.mkdir(model_Pathsave)
    tensorboard_Path = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/2018_AI_challenge_seg/Graph_cls_only_20180912_1'                ##Path to save tensorboard graph
    if os.path.exists(tensorboard_Path) == 0:
        os.mkdir(tensorboard_Path)

    ClsBase_model = all_model.cls_base_layers()
    Cls_model = all_model.cls()
    model = all_model.vgg_cls(ClsBase_model,Cls_model)     ##model construction
    model.load_weights('/media/root/3339482d-9d23-44ee-99a0-85e517217d15/2018_AI_challenge_seg/model/fit_cls_only_20180912_1/weights.08-0.06.hdf5')
    # model_multi_gpu = multi_gpu_model(model,gpus=gpus_used)   ##set multi gpu
    ##callbacks
    callback_checkpoint = ModelCheckpoint(filepath = model_Pathsave+"weights.{epoch:02d}-{val_loss:.2f}.hdf5",monitor='val_fmeasure',verbose=0,save_best_only=False,
                                          save_weights_only=False,mode='min',period=1)
    callback_tensorboard = TensorBoard(log_dir=tensorboard_Path,histogram_freq=0,write_graph=True,write_images=True)
    # callback_cycle_lr = SGDRScheduler(min_lr=1e-5,max_lr=1e-3,steps_per_epoch=np.ceil(trainset_num//batch_size), lr_decay=0.9, cycle_length=3,mult_factor=1.5)
    ##compile model
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy',metrics=['acc'])
    # model.load_weights('/media/root/3339482d-9d23-44ee-99a0-85e517217d15/2018_AI_challenge_seg/model/fit_cls_20180905_1/weights.200-0.17.hdf5',by_name=False)

    ##train model
    model.fit_generator(generate_classification_data_from_file(train_PathH5,batch_size=batch_size_cls),
                        validation_data=generate_classification_data_from_file_valid(evalu_PathH5,batch_size=1)
                        ,callbacks=[callback_tensorboard]
                        ,steps_per_epoch=trainset_num//batch_size_cls,epochs=1,validation_steps=evaluset_num,max_queue_size=1)

def main_test_cls():

    train_PathH5 = '/media/root/01456da2-1f1d-4b67-810e-b9cd3341133d/AI challenge/2018_seg/Edema_trainingset/'   ##train data , *.h5
    trainset_num = len(os.listdir(train_PathH5+'original_images/'))*128
    evalu_PathH5 = '/media/root/01456da2-1f1d-4b67-810e-b9cd3341133d/AI challenge/2018_seg/Edema_validationset/'##evaluate data , *.h5
    evaluset_num = len(os.listdir(evalu_PathH5+'original_images/')) * 128
    model_Pathsave = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/2018_AI_challenge_seg/model/fit_cls_only_20180912_1/'    ##Path to save model
    if os.path.exists(model_Pathsave) == 0:
        os.mkdir(model_Pathsave)
    tensorboard_Path = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/2018_AI_challenge_seg/Graph_cls_only_20180912_1'                ##Path to save tensorboard graph
    if os.path.exists(tensorboard_Path) == 0:
        os.mkdir(tensorboard_Path)

    ClsBase_model = all_model.cls_base_layers()
    Cls_model = all_model.cls()
    model = all_model.vgg_cls(ClsBase_model,Cls_model)     ##model construction
    model.load_weights('/media/root/3339482d-9d23-44ee-99a0-85e517217d15/2018_AI_challenge_seg/model/fit_cls_only_20180912_1/weights.10-0.30.hdf5')
    # model_multi_gpu = multi_gpu_model(model,gpus=gpus_used)   ##set multi gpu
    ##callbacks
    callback_checkpoint = ModelCheckpoint(filepath = model_Pathsave+"weights.{epoch:02d}-{val_loss:.2f}.hdf5",monitor='val_fmeasure',verbose=0,save_best_only=False,
                                          save_weights_only=False,mode='min',period=1)
    callback_tensorboard = TensorBoard(log_dir=tensorboard_Path,histogram_freq=0,write_graph=True,write_images=True)
    # callback_cycle_lr = SGDRScheduler(min_lr=1e-5,max_lr=1e-3,steps_per_epoch=np.ceil(trainset_num//batch_size), lr_decay=0.9, cycle_length=3,mult_factor=1.5)
    ##compile model
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy',metrics=['acc'])
    # model.load_weights('/media/root/3339482d-9d23-44ee-99a0-85e517217d15/2018_AI_challenge_seg/model/fit_cls_20180905_1/weights.200-0.17.hdf5',by_name=False)

    ##train model
    result = model.evaluate_generator(generate_classification_data_from_file_valid(evalu_PathH5,batch_size=1),max_queue_size=1,steps=evaluset_num,verbose=1)
    print(result)
## find the max and min learning rate
def main_find_lr():
    train_PathH5 = '/media/root/01456da2-1f1d-4b67-810e-b9cd3341133d/AI challenge/2018_seg/Edema_trainingset/'   ##train data , *.h5
    trainset_num = len(os.listdir(train_PathH5))*128
    tensorboard_Path = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/2018_AI_challenge_seg/Graph_lr'                ##Path to save tensorboard graph
    if os.path.exists(tensorboard_Path) == 0:
        os.mkdir(tensorboard_Path)

    model = all_model.Unet_class(input_shape=(image_x,image_y,3))     ##model construction

    ##callbacks
    callback_lr_finder = LRFinder(min_lr=1e-5, max_lr=1e-2, steps_per_epoch=np.ceil(trainset_num//batch_size), epochs=1)
    callback_tensorboard = TensorBoard(log_dir=tensorboard_Path, histogram_freq=0, write_graph=True, write_images=True)
    ##compile model
    model.compile(optimizer=Adam(), loss='binary_crossentropy',loss_weights=[1,0.5],
                         metrics=['acc', all_index.precision, all_index.recall, all_index.fmeasure, all_index.yt_sum])

    ##train model
    model.fit_generator(generate_segmentation_data_from_file(train_PathH5,batch_size=batch_size)
                        ,callbacks=[callback_lr_finder,callback_tensorboard]
                        ,steps_per_epoch=trainset_num//batch_size,epochs=10)
    # Plot the smoothed losses
    callback_lr_finder.plot_avg_loss()

def main_test_all():
    # time.sleep(14400)
    model_AllPathH5 = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/model/fit_seg_20181116_1/'
    model_list = os.listdir(model_AllPathH5)
    model_num = len(model_list)
    test_PathH5 = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/data/valid_1_old/'  ##path of saving test data, *.h5
    test_num = len(os.listdir(test_PathH5))
    dsc_max = 0
    best_model_name = 'JohnCena'
    for num in range(model_num):
        model_PathH5 = model_AllPathH5+model_list[num] ##model weight path
        model = load_model(model_PathH5,compile=False)  ##load model construction

        ##compile model
        model.compile(optimizer=Adam(lr=1.0e-5), loss='binary_crossentropy',
                             metrics=['acc', all_index.precision, all_index.recall, all_index.fmeasure, all_index.yt_sum,
                                      all_index.tp,all_index.tn,all_index.fp,all_index.fn])

        ##test model
        result = model.evaluate_generator(generate_segmentation_data_from_file_valid_t2(test_PathH5,batch_size=1),steps=test_num//1,verbose=1)

        ## cal recall,precision,dsc
        recall,precision,dsc = cal_index.cal_seg_index(result,test_num)
        print('recall:',recall,' precision:',precision,' dsc:',dsc,'\n')
        if dsc>dsc_max:
            dsc_max = dsc
            best_model_name = model_list[num]
    print('Best model:',best_model_name,' Best DSC:',dsc_max)

def main_test():
    model_PathH5 = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/model/fit_seg_20181206_3/weights.190-0.10.hdf5' ##model weight path
    model = load_model(model_PathH5,compile=False)  ##load model construction
    # image_savePath = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/image_show/20181008_1/'
    # if os.path.exists(image_savePath) == 0:
    #     os.mkdir(image_savePath)
    test_PathH5 = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/data/patient_2D_test/'   ##path of saving test data, *.h5
    # test_PathH5 = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/data/patient_test_person_2D/66/'  ##path of saving test data, *.h5
    test_num = len(os.listdir(test_PathH5))

    ##compile model
    model.compile(optimizer=Adam(lr=1.0e-5), loss='binary_crossentropy',
                         metrics=['acc', all_index.precision, all_index.recall, all_index.fmeasure, all_index.yt_sum,
                                  all_index.tp,all_index.tn,all_index.fp,all_index.fn])

    ##test model
    result = model.evaluate_generator(generate_segmentation_data_from_file_test_t2(test_PathH5,batch_size=1),steps=test_num//1,verbose=1)
    print(result)

    # ##test model on batch
    # tp = 0
    # tn = 0
    # fp = 0
    # fn = 0

    ## cal recall,precision,dsc
    recall,precision,dsc = cal_index.cal_seg_index(result,test_num)
    print('recall:',recall,' precision:',precision,' dsc:',dsc,'\n')

def main_test_bypredict():
    seg_thresh = 0.5
    model_PathH5 = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/model/fit_seg_20181128_2/weights.235-0.07.hdf5' ##model weight path
    model = load_model(model_PathH5,compile=False)  ##load model construction
    image_savePath = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/image_show/20181128_2_new/'
    if os.path.exists(image_savePath) == 0:
        os.mkdir(image_savePath)
    image_savePath_bad = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/image_show/20181128_2_new_bad/'
    if os.path.exists(image_savePath_bad) == 0:
        os.mkdir(image_savePath_bad)
    test_PathH5 = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/data/test_1_old/'   ##path of saving test data, *.h5
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

        # if np.sum(data_gt) == 0:
        #     continue

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

        if dsc<0.5:
            fig.savefig(image_savePath_bad + testList[num] + '_' + str(dsc) +'_.png')
            plt.close()
        else:
            fig.savefig(image_savePath + testList[num] + '_' + str(dsc) + '_.png')
            plt.close()

        ## cal all recall,precision,dsc
    recall, precision, dsc = cal_index.cal_rpd(tpall[target_num], tnall[target_num], fpall[target_num], fnall[target_num])
    print(target_num,' ALL recall:', recall, ' precision:', precision, ' dsc:', dsc)

if __name__ == "__main__":
    main_test()


