## Bin Huang, made in 2018.8.31

import random
import copy
import h5py
import cv2
from skimage import measure
from keras.models import *
from keras.layers import *
from PIL import Image

import scipy.ndimage

def generate_segmentation_data_from_file_t2_3D(path,batch_size=1):
    DataDirPath = path
    listpath = os.listdir(DataDirPath)

    dataall_num = len(listpath)

    listdataset_target = []
    listdataset_target.extend(listpath)
    for i in range(0,dataall_num):
        Path_datadir = DataDirPath
        listdataset_target[i] = Path_datadir + listpath[i]

    AllDataNum = len(listdataset_target)
    print('AllDataNum: ',AllDataNum)

    count_target = 0
    bc = 0
    while True:

        data_fullpath = listdataset_target[count_target]
        count_target+=1
        # print('target!')
        if count_target>= AllDataNum:
            count_target = 0
            random.shuffle(listdataset_target)

        f = h5py.File(data_fullpath)
        # t1_1 = f['t1'][:]
        # t1_1 = np.transpose(t1_1,[1,0,2])
        t2 = f['t2'][:]
        t2 = np.transpose(t2, [1, 0, 2,3])
        label = f['gt'][:]
        label = np.transpose(label, [1, 0, 2,3])
        f.close()
        # t1_1 = t1_1*1.0
        t2 = t2*1.0
        # print(np.max(t1_1),np.min(t1_1))
        # print(np.max(t2_1), np.min(t2_1))

        label_shape = np.shape(label)

        t2_1 = np.zeros([256,256,64],dtype='float')
        label_1 = np.zeros([256, 256, 64], dtype='float')

        t2_1[:,:,4:4+label_shape[2]] = t2[:,:,:,0]
        label_1[:, :, 4:4+label_shape[2]] = label[:, :, :,0]

        # ## only t1
        # t1_1 = cv2.resize(t1_1, (512, 512), interpolation=cv2.INTER_CUBIC)
        # t1_1 = np.expand_dims(t1_1, axis=3)

        flip_num = random.randint(-1, 2)
        if flip_num != 2:
            # t1_1 = cv2.flip(t1_1,1,dst=None) #1 shuiping, 0 chuizhi, -1 duijiao
            t2_1 = cv2.flip(t2_1, 1, dst=None)  # 1 shuiping, 0 chuizhi, -1 duijiao
            label_1 = cv2.flip(label_1,1,dst=None)
        #     t1_1 = np.expand_dims(t1_1,axis=3)
        t2_1 = np.expand_dims(t2_1, axis=4)
        label_1 = np.expand_dims(label_1, axis=4)
        #

        # t1_1 = np.expand_dims(t1_1, axis=0)
        t2_1 = np.expand_dims(t2_1, axis=0)
        label_1 = np.expand_dims(label_1, axis=0)

        if bc == 0:
            # data_t1 = copy.deepcopy(t1_1)
            data_t2 = copy.deepcopy(t2_1)
            label_train = copy.deepcopy(label_1)
        elif bc > 0:
            # data_t1 = np.concatenate([data_t1,t1_1],axis=0)
            data_t2 = np.concatenate([data_t2, t2_1], axis=0)
            label_train = np.concatenate([label_train, label_1], axis=0)

        bc+=1

        if bc>= batch_size:
            bc = 0
            yield [data_t2],label_train

def generate_segmentation_data_from_file_t1_3D(path,batch_size=1):
    DataDirPath = path
    listpath = os.listdir(DataDirPath)

    dataall_num = len(listpath)

    listdataset_target = []
    listdataset_target.extend(listpath)
    for i in range(0,dataall_num):
        Path_datadir = DataDirPath
        listdataset_target[i] = Path_datadir + listpath[i]

    AllDataNum = len(listdataset_target)
    print('AllDataNum: ',AllDataNum)

    count_target = 0
    bc = 0
    while True:

        data_fullpath = listdataset_target[count_target]
        count_target+=1
        # print('target!')
        if count_target>= AllDataNum:
            count_target = 0
            random.shuffle(listdataset_target)

        f = h5py.File(data_fullpath)
        # t1_1 = f['t1'][:]
        # t1_1 = np.transpose(t1_1,[1,0,2])
        t2 = f['rt1'][:]
        t2 = np.transpose(t2, [1, 0, 2,3])
        label = f['gt'][:]
        label = np.transpose(label, [1, 0, 2,3])
        f.close()
        # t1_1 = t1_1*1.0
        t2 = t2*1.0
        # print(np.max(t1_1),np.min(t1_1))
        # print(np.max(t2_1), np.min(t2_1))

        label_shape = np.shape(label)

        t2_1 = np.zeros([256,256,64],dtype='float')
        label_1 = np.zeros([256, 256, 64], dtype='float')

        t2_1[:,:,4:4+label_shape[2]] = t2[:,:,:,0]
        label_1[:, :, 4:4+label_shape[2]] = label[:, :, :,0]

        # ## only t1
        # t1_1 = cv2.resize(t1_1, (512, 512), interpolation=cv2.INTER_CUBIC)
        # t1_1 = np.expand_dims(t1_1, axis=3)

        flip_num = random.randint(-1, 2)
        if flip_num != 2:
            # t1_1 = cv2.flip(t1_1,1,dst=None) #1 shuiping, 0 chuizhi, -1 duijiao
            t2_1 = cv2.flip(t2_1, 1, dst=None)  # 1 shuiping, 0 chuizhi, -1 duijiao
            label_1 = cv2.flip(label_1,1,dst=None)
        #     t1_1 = np.expand_dims(t1_1,axis=3)
        t2_1 = np.expand_dims(t2_1, axis=4)
        label_1 = np.expand_dims(label_1, axis=4)
        #

        # t1_1 = np.expand_dims(t1_1, axis=0)
        t2_1 = np.expand_dims(t2_1, axis=0)
        label_1 = np.expand_dims(label_1, axis=0)

        if bc == 0:
            # data_t1 = copy.deepcopy(t1_1)
            data_t2 = copy.deepcopy(t2_1)
            label_train = copy.deepcopy(label_1)
        elif bc > 0:
            # data_t1 = np.concatenate([data_t1,t1_1],axis=0)
            data_t2 = np.concatenate([data_t2, t2_1], axis=0)
            label_train = np.concatenate([label_train, label_1], axis=0)

        bc+=1

        if bc>= batch_size:
            bc = 0
            yield [data_t2],label_train

def generate_segmentation_data_from_file_multi_3D(path,batch_size=1,augment=True):
    DataDirPath = path
    listpath = os.listdir(DataDirPath)

    dataall_num = len(listpath)

    listdataset_target = []
    listdataset_target.extend(listpath)
    for i in range(0,dataall_num):
        Path_datadir = DataDirPath
        listdataset_target[i] = Path_datadir + listpath[i]

    AllDataNum = len(listdataset_target)
    print('AllDataNum: ',AllDataNum)

    count_target = 0
    bc = 0
    while True:

        data_fullpath = listdataset_target[count_target]
        count_target+=1
        # print('target!')
        if count_target>= AllDataNum:
            count_target = 0
            random.shuffle(listdataset_target)

        f = h5py.File(data_fullpath)
        t1 = f['rt1'][:]
        t1 = np.transpose(t1,[1,0,2,3])
        t2 = f['t2'][:]
        t2 = np.transpose(t2, [1, 0, 2,3])
        label = f['gt'][:]
        label = np.transpose(label, [1, 0, 2,3])
        f.close()

        max_t1 = np.max(t1)
        min_t1 = np.min(t1)
        max_t2 = np.max(t2)
        min_t2 = np.min(t2)

        t1 = (t1-min_t1)/(max_t1-min_t1)
        t2 = (t2 - min_t2) / (max_t2 - min_t2)

        # t1 = t1*1.0
        # t2 = t2*1.0
        # print(np.max(t1_1),np.min(t1_1))
        # print(np.max(t2_1), np.min(t2_1))

        label_shape = np.shape(label)

        t1_1 = np.zeros([256, 256, 64], dtype='float')
        t2_1 = np.zeros([256,256,64],dtype='float')
        label_1 = np.zeros([256, 256, 64], dtype='float')

        t1_1[:, :, 4:4 + label_shape[2]] = t1[:, :, :, 0]
        t2_1[:,:,4:4+label_shape[2]] = t2[:,:,:,0]
        label_1[:, :, 4:4+label_shape[2]] = label[:, :, :,0]


        if augment:
            flip_num = random.randint(-1, 2)
            if flip_num != 2:
                t1_1 = cv2.flip(t1_1,1,dst=None) #1 shuiping, 0 chuizhi, -1 duijiao
                t2_1 = cv2.flip(t2_1, 1, dst=None)  # 1 shuiping, 0 chuizhi, -1 duijiao
                label_1 = cv2.flip(label_1,1,dst=None)
            t1_1 = np.expand_dims(t1_1,axis=4)
            t2_1 = np.expand_dims(t2_1, axis=4)
            label_1 = np.expand_dims(label_1, axis=4)

            zoom_num = random.choice([0.75,1,1.25])
            if zoom_num != 1:
                t1_2 = copy.deepcopy(t1_1)
                t2_2 = copy.deepcopy(t2_1)
                label_2 = copy.deepcopy(label_1)
                t1_2 = scipy.ndimage.zoom(t1_2,zoom_num,order=3)
                t2_2 = scipy.ndimage.zoom(t2_2, zoom_num, order=3)
                label_2 = scipy.ndimage.zoom(label_2, zoom_num, order=3)
                label_2[label_2>=0.02] = 1
                label_2[label_2 < 0.02] = 0

                t1_1 = np.zeros([256, 256, 64, 1], dtype='float')
                t2_1 = np.zeros([256, 256, 64, 1], dtype='float')
                label_1 = np.zeros([256, 256, 64, 1], dtype='float')
                if zoom_num == 1.25:
                    t1_1[:,:,:,:] = t1_2[32:288,32:288,8:72,:]
                    t2_1[:, :, :, :] = t2_2[32:288, 32:288, 8:72, :]
                    label_1[:, :, :, :] = label_2[32:288, 32:288, 8:72, :]
                elif zoom_num == 0.75:
                    t1_1[32:224, 32:224, 8:56, :] = t1_2[:,:,:,:]
                    t2_1[32:224, 32:224, 8:56, :] = t2_2[:,:,:,:]
                    label_1[32:224, 32:224, 8:56, :] = label_2[:,:,:,:]

        t1_1 = np.expand_dims(t1_1, axis=0)
        t2_1 = np.expand_dims(t2_1, axis=0)
        label_1 = np.expand_dims(label_1, axis=0)

        if bc == 0:
            data_t1 = copy.deepcopy(t1_1)
            data_t2 = copy.deepcopy(t2_1)
            label_train = copy.deepcopy(label_1)
        elif bc > 0:
            data_t1 = np.concatenate([data_t1,t1_1],axis=0)
            data_t2 = np.concatenate([data_t2, t2_1], axis=0)
            label_train = np.concatenate([label_train, label_1], axis=0)

        bc+=1

        if bc>= batch_size:
            bc = 0
            yield [data_t2,data_t1],[label_train]

def generate_segmentation_data_from_file_multi_3D_multiout(path,batch_size=1,augment=True):
    DataDirPath = path
    listpath = os.listdir(DataDirPath)

    dataall_num = len(listpath)

    listdataset_target = []
    listdataset_target.extend(listpath)
    for i in range(0,dataall_num):
        Path_datadir = DataDirPath
        listdataset_target[i] = Path_datadir + listpath[i]

    AllDataNum = len(listdataset_target)
    print('AllDataNum: ',AllDataNum)

    count_target = 0
    bc = 0
    while True:

        data_fullpath = listdataset_target[count_target]
        count_target+=1
        # print('target!')
        if count_target>= AllDataNum:
            count_target = 0
            random.shuffle(listdataset_target)

        f = h5py.File(data_fullpath)
        t1 = f['t1'][:]
        t1 = np.transpose(t1,[1,0,2,3])
        t2 = f['t2'][:]
        t2 = np.transpose(t2, [1, 0, 2,3])
        label = f['gt'][:]
        label = np.transpose(label, [1, 0, 2,3])
        label_nx = f['gt_nx'][:]
        label_nx = np.transpose(label_nx, [1, 0, 2,3])
        f.close()

        max_t1 = np.max(t1)
        min_t1 = np.min(t1)
        max_t2 = np.max(t2)
        min_t2 = np.min(t2)

        t1 = (t1-min_t1)/(max_t1-min_t1)
        t2 = (t2 - min_t2) / (max_t2 - min_t2)

        # t1 = t1*1.0
        # t2 = t2*1.0
        # print(np.max(t1_1),np.min(t1_1))
        # print(np.max(t2_1), np.min(t2_1))

        label_shape = np.shape(label)

        t1_1 = np.zeros([256, 256, 64], dtype='float')
        t2_1 = np.zeros([256,256,64],dtype='float')
        label_1 = np.zeros([256, 256, 64], dtype='float')
        label_nx_1 = np.zeros([256, 256, 64], dtype='float')

        t1_1[:, :, 4:4 + label_shape[2]] = t1[:, :, :, 0]
        t2_1[:,:,4:4+label_shape[2]] = t2[:,:,:,0]
        label_1[:, :, 4:4+label_shape[2]] = label[:, :, :,0]
        label_nx_1[:, :, 4:4 + label_shape[2]] = label_nx[:, :, :, 0]


        if augment:
            flip_num = random.randint(-1, 2)
            if flip_num != 2:
                t1_1 = cv2.flip(t1_1,1,dst=None) #1 shuiping, 0 chuizhi, -1 duijiao
                t2_1 = cv2.flip(t2_1, 1, dst=None)  # 1 shuiping, 0 chuizhi, -1 duijiao
                label_1 = cv2.flip(label_1,1,dst=None)
                label_nx_1 = cv2.flip(label_nx_1, 1, dst=None)
            t1_1 = np.expand_dims(t1_1,axis=4)
            t2_1 = np.expand_dims(t2_1, axis=4)
            label_1 = np.expand_dims(label_1, axis=4)
            label_nx_1 = np.expand_dims(label_nx_1, axis=4)

            zoom_num = random.choice([0.75,1,1.25])
            if zoom_num != 1:
                t1_2 = copy.deepcopy(t1_1)
                t2_2 = copy.deepcopy(t2_1)
                label_2 = copy.deepcopy(label_1)
                label_nx_2 = copy.deepcopy(label_nx_1)
                t1_2 = scipy.ndimage.zoom(t1_2,zoom_num,order=3)
                t2_2 = scipy.ndimage.zoom(t2_2, zoom_num, order=3)
                label_2 = scipy.ndimage.zoom(label_2, zoom_num, order=3)
                label_2[label_2>=0.05] = 1
                label_2[label_2 < 0.05] = 0
                label_nx_2 = scipy.ndimage.zoom(label_nx_2, zoom_num, order=3)
                label_nx_2[label_nx_2>=0.05] = 1
                label_nx_2[label_nx_2 < 0.05] = 0

                t1_1 = np.zeros([256, 256, 64, 1], dtype='float')
                t2_1 = np.zeros([256, 256, 64, 1], dtype='float')
                label_1 = np.zeros([256, 256, 64, 1], dtype='float')
                label_nx_1 = np.zeros([256, 256, 64, 1], dtype='float')
                if zoom_num == 1.25:
                    t1_1[:,:,:,:] = t1_2[32:288,32:288,8:72,:]
                    t2_1[:, :, :, :] = t2_2[32:288, 32:288, 8:72, :]
                    label_1[:, :, :, :] = label_2[32:288, 32:288, 8:72, :]
                    label_nx_1[:, :, :, :] = label_nx_2[32:288, 32:288, 8:72, :]
                elif zoom_num == 0.75:
                    t1_1[32:224, 32:224, 8:56, :] = t1_2[:,:,:,:]
                    t2_1[32:224, 32:224, 8:56, :] = t2_2[:,:,:,:]
                    label_1[32:224, 32:224, 8:56, :] = label_2[:,:,:,:]
                    label_nx_1[32:224, 32:224, 8:56, :] = label_nx_2[:, :, :, :]

        t1_1 = np.expand_dims(t1_1, axis=0)
        t2_1 = np.expand_dims(t2_1, axis=0)
        label_1 = np.expand_dims(label_1, axis=0)
        label_nx_1 = np.expand_dims(label_nx_1, axis=0)
        label_nd_1 = label_1 - label_nx_1
        label_1 = label_nd_1

        if bc == 0:
            data_t1 = copy.deepcopy(t1_1)
            data_t2 = copy.deepcopy(t2_1)
            label_train = copy.deepcopy(label_1)
            label_nx_train = copy.deepcopy(label_nx_1)
        elif bc > 0:
            data_t1 = np.concatenate([data_t1,t1_1],axis=0)
            data_t2 = np.concatenate([data_t2, t2_1], axis=0)
            label_train = np.concatenate([label_train, label_1], axis=0)
            label_nx_train = np.concatenate([label_nx_train, label_nx_1], axis=0)

        bc+=1

        if bc>= batch_size:
            bc = 0
            yield [data_t2,data_t1],[label_train,label_nx_train]

def generate_segmentation_data_from_file_valid_t2_3D(path,batch_size=1):
    DataDirPath = path
    listpath = os.listdir(DataDirPath)

    dataall_num = len(listpath)

    listdataset_target = []
    listdataset_target.extend(listpath)
    for i in range(0,dataall_num):
        Path_datadir = DataDirPath
        listdataset_target[i] = Path_datadir + listpath[i]

    AllDataNum = len(listdataset_target)
    print('AllDataNum: ', AllDataNum)

    count_target = 0
    bc = 0

    while True:

        data_fullpath = listdataset_target[count_target]
        count_target += 1
        if count_target >= AllDataNum:
            count_target = 0

        f = h5py.File(data_fullpath)
        # t1_1 = f['t1'][:]
        # t1_1 = np.transpose(t1_1,[1,0,2])
        t2 = f['t2'][:]
        t2 = np.transpose(t2, [1, 0, 2,3])
        label = f['gt'][:]
        label = np.transpose(label, [1, 0, 2,3])
        f.close()
        # t1_1 = t1_1*1.0
        t2 = t2*1.0
        # print(np.max(t1_1),np.min(t1_1))
        # print(np.max(t2_1), np.min(t2_1))

        label_shape = np.shape(label)

        t2_1 = np.zeros([256,256,64],dtype='float')
        label_1 = np.zeros([256, 256, 64], dtype='float')

        t2_1[:,:,4:4+label_shape[2]] = t2[:,:,:,0]
        label_1[:, :, 4:4+label_shape[2]] = label[:, :, :,0]

        t2_1 = np.expand_dims(t2_1, axis=4)
        label_1 = np.expand_dims(label_1, axis=4)

        # t1_1 = np.expand_dims(t1_1, axis=0)
        t2_1 = np.expand_dims(t2_1, axis=0)
        label_1 = np.expand_dims(label_1, axis=0)



        if bc == 0:
            # data_t1 = copy.deepcopy(t1_1)
            data_t2 = copy.deepcopy(t2_1)
            label_train = copy.deepcopy(label_1)
        elif bc > 0:
            # data_t1 = np.concatenate([data_t1, t1_1], axis=0)
            data_t2 = np.concatenate([data_t2, t2_1], axis=0)
            label_train = np.concatenate([label_train, label_1], axis=0)

        bc += 1

        if bc >= batch_size:
            bc = 0
            yield [data_t2], label_train

def generate_segmentation_data_from_file_valid_t1_3D(path,batch_size=1):
    DataDirPath = path
    listpath = os.listdir(DataDirPath)

    dataall_num = len(listpath)

    listdataset_target = []
    listdataset_target.extend(listpath)
    for i in range(0,dataall_num):
        Path_datadir = DataDirPath
        listdataset_target[i] = Path_datadir + listpath[i]

    AllDataNum = len(listdataset_target)
    print('AllDataNum: ', AllDataNum)

    count_target = 0
    bc = 0

    while True:

        data_fullpath = listdataset_target[count_target]
        count_target += 1
        if count_target >= AllDataNum:
            count_target = 0

        f = h5py.File(data_fullpath)
        # t1_1 = f['t1'][:]
        # t1_1 = np.transpose(t1_1,[1,0,2])
        t2 = f['rt1'][:]
        t2 = np.transpose(t2, [1, 0, 2,3])
        label = f['gt'][:]
        label = np.transpose(label, [1, 0, 2,3])
        f.close()
        # t1_1 = t1_1*1.0
        t2 = t2*1.0
        # print(np.max(t1_1),np.min(t1_1))
        # print(np.max(t2_1), np.min(t2_1))

        label_shape = np.shape(label)

        t2_1 = np.zeros([256,256,64],dtype='float')
        label_1 = np.zeros([256, 256, 64], dtype='float')

        t2_1[:,:,4:4+label_shape[2]] = t2[:,:,:,0]
        label_1[:, :, 4:4+label_shape[2]] = label[:, :, :,0]

        t2_1 = np.expand_dims(t2_1, axis=4)
        label_1 = np.expand_dims(label_1, axis=4)

        # t1_1 = np.expand_dims(t1_1, axis=0)
        t2_1 = np.expand_dims(t2_1, axis=0)
        label_1 = np.expand_dims(label_1, axis=0)



        if bc == 0:
            # data_t1 = copy.deepcopy(t1_1)
            data_t2 = copy.deepcopy(t2_1)
            label_train = copy.deepcopy(label_1)
        elif bc > 0:
            # data_t1 = np.concatenate([data_t1, t1_1], axis=0)
            data_t2 = np.concatenate([data_t2, t2_1], axis=0)
            label_train = np.concatenate([label_train, label_1], axis=0)

        bc += 1

        if bc >= batch_size:
            bc = 0
            yield [data_t2], label_train

def generate_segmentation_data_from_file_valid_multi_3D(path,batch_size=1):
    DataDirPath = path
    listpath = os.listdir(DataDirPath)

    dataall_num = len(listpath)

    listdataset_target = []
    listdataset_target.extend(listpath)
    for i in range(0,dataall_num):
        Path_datadir = DataDirPath
        listdataset_target[i] = Path_datadir + listpath[i]

    AllDataNum = len(listdataset_target)
    print('AllDataNum: ', AllDataNum)

    count_target = 0
    bc = 0

    while True:

        data_fullpath = listdataset_target[count_target]
        count_target += 1
        if count_target >= AllDataNum:
            count_target = 0

        f = h5py.File(data_fullpath)
        t1 = f['rt1'][:]
        t1 = np.transpose(t1,[1,0,2,3])
        t2 = f['t2'][:]
        t2 = np.transpose(t2, [1, 0, 2,3])
        label = f['gt'][:]
        label = np.transpose(label, [1, 0, 2,3])
        f.close()

        max_t1 = np.max(t1)
        min_t1 = np.min(t1)
        max_t2 = np.max(t2)
        min_t2 = np.min(t2)

        t1 = (t1-min_t1)/(max_t1-min_t1)
        t2 = (t2 - min_t2) / (max_t2 - min_t2)

        t1 = t1*1.0
        t2 = t2*1.0
        # print(np.max(t1_1),np.min(t1_1))
        # print(np.max(t2_1), np.min(t2_1))

        label_shape = np.shape(label)

        t1_1 = np.zeros([256, 256, 64], dtype='float')
        t2_1 = np.zeros([256,256,64],dtype='float')
        label_1 = np.zeros([256, 256, 64], dtype='float')

        t1_1[:, :, 4:4 + label_shape[2]] = t1[:, :, :, 0]
        t2_1[:,:,4:4+label_shape[2]] = t2[:,:,:,0]
        label_1[:, :, 4:4+label_shape[2]] = label[:, :, :,0]

        t1_1 = np.expand_dims(t1_1, axis=4)
        t2_1 = np.expand_dims(t2_1, axis=4)
        label_1 = np.expand_dims(label_1, axis=4)

        t1_1 = np.expand_dims(t1_1, axis=0)
        t2_1 = np.expand_dims(t2_1, axis=0)
        label_1 = np.expand_dims(label_1, axis=0)



        if bc == 0:
            data_t1 = copy.deepcopy(t1_1)
            data_t2 = copy.deepcopy(t2_1)
            label_train = copy.deepcopy(label_1)
        elif bc > 0:
            data_t1 = np.concatenate([data_t1, t1_1], axis=0)
            data_t2 = np.concatenate([data_t2, t2_1], axis=0)
            label_train = np.concatenate([label_train, label_1], axis=0)

        bc += 1

        if bc >= batch_size:
            bc = 0
            yield [data_t2,data_t1], [label_train]

def generate_segmentation_data_from_file_valid_multi_3D_multiout(path,batch_size=1):
    DataDirPath = path
    listpath = os.listdir(DataDirPath)

    dataall_num = len(listpath)

    listdataset_target = []
    listdataset_target.extend(listpath)
    for i in range(0,dataall_num):
        Path_datadir = DataDirPath
        listdataset_target[i] = Path_datadir + listpath[i]

    AllDataNum = len(listdataset_target)
    print('AllDataNum: ', AllDataNum)

    count_target = 0
    bc = 0

    while True:

        data_fullpath = listdataset_target[count_target]
        count_target += 1
        if count_target >= AllDataNum:
            count_target = 0

        f = h5py.File(data_fullpath)
        t1 = f['t1'][:]
        t1 = np.transpose(t1,[1,0,2,3])
        t2 = f['t2'][:]
        t2 = np.transpose(t2, [1, 0, 2,3])
        label = f['gt'][:]
        label = np.transpose(label, [1, 0, 2,3])
        label_nx = f['gt_nx'][:]
        label_nx = np.transpose(label_nx, [1, 0, 2, 3])
        f.close()

        max_t1 = np.max(t1)
        min_t1 = np.min(t1)
        max_t2 = np.max(t2)
        min_t2 = np.min(t2)

        t1 = (t1-min_t1)/(max_t1-min_t1)
        t2 = (t2 - min_t2) / (max_t2 - min_t2)

        t1 = t1*1.0
        t2 = t2*1.0
        # print(np.max(t1_1),np.min(t1_1))
        # print(np.max(t2_1), np.min(t2_1))

        label_shape = np.shape(label)

        t1_1 = np.zeros([256, 256, 64], dtype='float')
        t2_1 = np.zeros([256,256,64],dtype='float')
        label_1 = np.zeros([256, 256, 64], dtype='float')
        label_nx_1 = np.zeros([256, 256, 64], dtype='float')

        t1_1[:, :, 4:4 + label_shape[2]] = t1[:, :, :, 0]
        t2_1[:,:,4:4+label_shape[2]] = t2[:,:,:,0]
        label_1[:, :, 4:4+label_shape[2]] = label[:, :, :,0]
        label_nx_1[:, :, 4:4 + label_shape[2]] = label_nx[:, :, :, 0]

        t1_1 = np.expand_dims(t1_1, axis=4)
        t2_1 = np.expand_dims(t2_1, axis=4)
        label_1 = np.expand_dims(label_1, axis=4)
        label_nx_1 = np.expand_dims(label_nx_1, axis=4)

        t1_1 = np.expand_dims(t1_1, axis=0)
        t2_1 = np.expand_dims(t2_1, axis=0)
        label_1 = np.expand_dims(label_1, axis=0)
        label_nx_1 = np.expand_dims(label_nx_1, axis=0)
        label_nd_1 = label_1 - label_nx_1
        label_1 = label_nd_1

        if bc == 0:
            data_t1 = copy.deepcopy(t1_1)
            data_t2 = copy.deepcopy(t2_1)
            label_train = copy.deepcopy(label_1)
            label_nx_train = copy.deepcopy(label_nx_1)
        elif bc > 0:
            data_t1 = np.concatenate([data_t1, t1_1], axis=0)
            data_t2 = np.concatenate([data_t2, t2_1], axis=0)
            label_train = np.concatenate([label_train, label_1], axis=0)
            label_nx_train = np.concatenate([label_nx_train, label_nx_1], axis=0)

        bc += 1

        if bc >= batch_size:
            bc = 0
            yield [data_t2,data_t1], [label_train,label_nx_train]