## Bin Huang, made in 2018.8.31

import random
import copy
import h5py
import cv2
from skimage import measure
from keras.models import *
from keras.layers import *
from PIL import Image

def generate_segmentation_data_from_file(path,batch_size=1):
    DataDirPath = path
    listpath = os.listdir(DataDirPath)

    dataall_num = len(listpath)

    listdataset_target = []
    listdataset_notarget = []
    listdataset_target.extend(listpath)
    for i in range(0,dataall_num):
        Path_datadir = DataDirPath
        listdataset_target[i] = Path_datadir + listpath[i]

    AllDataNum = len(listdataset_target)
    print('AllDataNum: ',AllDataNum)
    pop_flag = 0
    for num in range(AllDataNum)[::-1]:
        data_path_read = copy.deepcopy(listdataset_target[num])
        f = h5py.File(data_path_read)
        label_0 = f['gt'][:]
        f.close()
        label_0[label_0 > 0] = 1
        if np.sum(label_0) == 0:
            listdataset_notarget.append(data_path_read)
            if pop_flag==0:
                listdataset_target.pop(num)
                pop_flag=1
        pop_flag = 0

    AllTargetNum = len(listdataset_target)
    AllNoTargetNum = len(listdataset_notarget)

    print('AllTargetNum:',AllTargetNum,' AllNoTargetNum:',AllNoTargetNum)


    random.shuffle(listdataset_target)

    random.shuffle(listdataset_notarget)

    cccc = 0
    count_num = 0
    count_target = 0
    count_notarget = 0
    bc = 0
    while True:

        if count_num<5:
            data_fullpath = listdataset_target[count_target]
            count_target+=1
            # print('target!')
            if count_target>= AllTargetNum:
                count_target = 0
                random.shuffle(listdataset_target)
        elif count_num >= 5:
            data_fullpath = listdataset_notarget[count_notarget]
            count_notarget += 1
            # print('notarget!')
            count_num = 0
            if count_notarget >= AllNoTargetNum:
                count_notarget = 0
                random.shuffle(listdataset_notarget)

        f = h5py.File(data_fullpath)
        t1_1 = f['t1'][:]
        t1_1 = np.transpose(t1_1,[1,0,2])
        t2_1 = f['t2'][:]
        t2_1 = np.transpose(t2_1, [1, 0, 2])
        label_1 = f['gt'][:]
        label_1 = np.transpose(label_1, [1, 0, 2])
        f.close()
        t1_1 = t1_1*1.0
        t2_1 = t2_1*1.0
        # print(np.max(t1_1),np.min(t1_1))
        # print(np.max(t2_1), np.min(t2_1))

        # ## only t1
        # t1_1 = cv2.resize(t1_1, (512, 512), interpolation=cv2.INTER_CUBIC)
        # t1_1 = np.expand_dims(t1_1, axis=3)

        flip_num = random.randint(0, 1)
        if flip_num == 1:
            t1_1 = cv2.flip(t1_1,1,dst=None) #1 shuiping, 0 chuizhi, -1 duijiao
            t2_1 = cv2.flip(t2_1, 1, dst=None)  # 1 shuiping, 0 chuizhi, -1 duijiao
            label_1 = cv2.flip(label_1,1,dst=None)
            t1_1 = np.expand_dims(t1_1,axis=3)
            t2_1 = np.expand_dims(t2_1, axis=3)
            label_1 = np.expand_dims(label_1, axis=3)

        resize_num = random.randint(0,2)
        if resize_num == 1:
            t1_1_r = cv2.resize(t1_1,(192,192),interpolation=cv2.INTER_CUBIC)
            t1_1 = np.zeros_like(t1_1)
            t1_1[32:224,32:224,0] = t1_1_r
            t2_1_r = cv2.resize(t2_1, (384, 384), interpolation=cv2.INTER_CUBIC)
            t2_1 = np.zeros_like(t2_1)
            t2_1[64:448,64:448,0] = t2_1_r
            label_1_r = cv2.resize(label_1, (384, 384), interpolation=cv2.INTER_CUBIC)
            label_1 = np.zeros_like(label_1)
            label_1[64:448, 64:448,0] = label_1_r
            label_1[label_1 > 0.2] = 1
            label_1[label_1 <= 0.2] = 0
        elif resize_num == 2:
            t1_1_r = cv2.resize(t1_1, (320, 320), interpolation=cv2.INTER_CUBIC)
            t1_1 = np.zeros_like(t1_1)
            t1_1[:,:,0] = t1_1_r[32:288,32:288]
            t2_1_r = cv2.resize(t2_1, (640, 640), interpolation=cv2.INTER_CUBIC)
            t2_1 = np.zeros_like(t2_1)
            t2_1[:,:,0] = t2_1_r[64:576,64:576]
            label_1_r = cv2.resize(label_1, (640, 640), interpolation=cv2.INTER_CUBIC)
            label_1 = np.zeros_like(label_1)
            label_1[:,:,0] = label_1_r[64:576, 64:576]
            label_1[label_1 > 0.2] = 1
            label_1[label_1 <= 0.2] = 0

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
        count_num+=1

        ## test picture
        # cccc+=1
        # image_save_path = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/image_show/'
        # t2_s = t2_1[0,:,:,0]
        # t1_s = t1_1[0,:,:,0]
        # label_s = label_1[0,:,:,0]
        # im_label = np.concatenate((t2_s,label_s),axis=1)
        # im_last = im_label*255
        # im_t1 = t1_s*255
        # new_im = Image.fromarray(im_last.astype(np.uint8))
        # new_im_t1 = Image.fromarray(im_t1.astype(np.uint8))
        # new_im.save(image_save_path+str(cccc)+'.jpg')
        # new_im_t1.save(image_save_path + str(cccc) + '_t1.jpg')

        if bc>= batch_size:
            bc = 0
            yield [data_t2,data_t1],label_train

def generate_segmentation_data_from_file_t2(path,batch_size=1):
    DataDirPath = path
    listpath = os.listdir(DataDirPath)

    dataall_num = len(listpath)

    listdataset_target = []
    listdataset_notarget = []
    listdataset_target.extend(listpath)
    for i in range(0,dataall_num):
        Path_datadir = DataDirPath
        listdataset_target[i] = Path_datadir + listpath[i]

    AllDataNum = len(listdataset_target)
    print('AllDataNum: ',AllDataNum)
    pop_flag = 0
    for num in range(AllDataNum)[::-1]:
        data_path_read = copy.deepcopy(listdataset_target[num])
        f = h5py.File(data_path_read)
        label_0 = f['gt'][:]
        f.close()
        label_0[label_0 > 0] = 1
        if np.sum(label_0) == 0:
            listdataset_notarget.append(data_path_read)
            if pop_flag==0:
                listdataset_target.pop(num)
                pop_flag=1
        pop_flag = 0

    AllTargetNum = len(listdataset_target)
    AllNoTargetNum = len(listdataset_notarget)

    print('AllTargetNum:',AllTargetNum,' AllNoTargetNum:',AllNoTargetNum)


    random.shuffle(listdataset_target)

    random.shuffle(listdataset_notarget)

    cccc = 0
    count_num = 0
    count_target = 0
    count_notarget = 0
    bc = 0
    while True:

        if count_num<5:
            data_fullpath = listdataset_target[count_target]
            count_target+=1
            # print('target!')
            if count_target>= AllTargetNum:
                count_target = 0
                random.shuffle(listdataset_target)
        elif count_num >= 5:
            data_fullpath = listdataset_notarget[count_notarget]
            count_notarget += 1
            # print('notarget!')
            count_num = 0
            if count_notarget >= AllNoTargetNum:
                count_notarget = 0
                random.shuffle(listdataset_notarget)

        f = h5py.File(data_fullpath)
        # t1_1 = f['t1'][:]
        # t1_1 = np.transpose(t1_1,[1,0,2])
        t2_1 = f['t2'][:]
        t2_1 = np.transpose(t2_1, [1, 0, 2])
        label_1 = f['gt'][:]
        label_1 = np.transpose(label_1, [1, 0, 2])
        f.close()
        # t1_1 = t1_1*1.0
        t2_1 = t2_1*1.0
        # print(np.max(t1_1),np.min(t1_1))
        # print(np.max(t2_1), np.min(t2_1))

        # ## only t1
        # t1_1 = cv2.resize(t1_1, (512, 512), interpolation=cv2.INTER_CUBIC)
        # t1_1 = np.expand_dims(t1_1, axis=3)

        flip_num = random.randint(-1, 2)
        if flip_num != 2:
            # t1_1 = cv2.flip(t1_1,1,dst=None) #1 shuiping, 0 chuizhi, -1 duijiao
            t2_1 = cv2.flip(t2_1, 1, dst=None)  # 1 shuiping, 0 chuizhi, -1 duijiao
            label_1 = cv2.flip(label_1,1,dst=None)
        #     t1_1 = np.expand_dims(t1_1,axis=3)
            t2_1 = np.expand_dims(t2_1, axis=3)
            label_1 = np.expand_dims(label_1, axis=3)
        #
        resize_num = random.randint(0,2)
        if resize_num == 1:
        #     t1_1_r = cv2.resize(t1_1,(192,192),interpolation=cv2.INTER_CUBIC)
        #     t1_1 = np.zeros_like(t1_1)
        #     t1_1[32:224,32:224,0] = t1_1_r
            t2_1_r = cv2.resize(t2_1, (384, 384), interpolation=cv2.INTER_CUBIC)
            t2_1 = np.zeros_like(t2_1)
            t2_1[64:448,64:448,0] = t2_1_r
            label_1_r = cv2.resize(label_1, (384, 384), interpolation=cv2.INTER_CUBIC)
            label_1 = np.zeros_like(label_1)
            label_1[64:448, 64:448,0] = label_1_r
            label_1[label_1 > 0.2] = 1
            label_1[label_1 <= 0.2] = 0
        elif resize_num == 2:
            # t1_1_r = cv2.resize(t1_1, (320, 320), interpolation=cv2.INTER_CUBIC)
        #     t1_1 = np.zeros_like(t1_1)
        #     t1_1[:,:,0] = t1_1_r[32:288,32:288]
            t2_1_r = cv2.resize(t2_1, (640, 640), interpolation=cv2.INTER_CUBIC)
            t2_1 = np.zeros_like(t2_1)
            t2_1[:,:,0] = t2_1_r[64:576,64:576]
            label_1_r = cv2.resize(label_1, (640, 640), interpolation=cv2.INTER_CUBIC)
            label_1 = np.zeros_like(label_1)
            label_1[:,:,0] = label_1_r[64:576, 64:576]
            label_1[label_1 > 0.2] = 1
            label_1[label_1 <= 0.2] = 0

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
        # count_num+=1
        #
        ## test picture
        # cccc+=1
        # image_save_path = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/image_show/'
        # t2_s = t2_1[0,:,:,0]
        # t1_s = t1_1[0,:,:,0]
        # label_s = label_1[0,:,:,0]
        # im_label = np.concatenate((t2_s,label_s),axis=1)
        # im_last = im_label*255
        # im_t1 = t1_s*255
        # new_im = Image.fromarray(im_last.astype(np.uint8))
        # new_im_t1 = Image.fromarray(im_t1.astype(np.uint8))
        # new_im.save(image_save_path+str(cccc)+'.jpg')
        # new_im_t1.save(image_save_path + str(cccc) + '_t1.jpg')

        if bc>= batch_size:
            bc = 0
            yield [data_t2],label_train

def generate_segmentation_data_from_file_t2_256(path,batch_size=1):
    DataDirPath = path
    listpath = os.listdir(DataDirPath)

    dataall_num = len(listpath)

    listdataset_target = []
    listdataset_notarget = []
    listdataset_target.extend(listpath)
    for i in range(0,dataall_num):
        Path_datadir = DataDirPath
        listdataset_target[i] = Path_datadir + listpath[i]

    AllDataNum = len(listdataset_target)
    print('AllDataNum: ',AllDataNum)
    pop_flag = 0
    for num in range(AllDataNum)[::-1]:
        data_path_read = copy.deepcopy(listdataset_target[num])
        f = h5py.File(data_path_read)
        label_0 = f['gt'][:]
        f.close()
        label_0[label_0 > 0] = 1
        if np.sum(label_0) == 0:
            listdataset_notarget.append(data_path_read)
            if pop_flag==0:
                listdataset_target.pop(num)
                pop_flag=1
        pop_flag = 0

    AllTargetNum = len(listdataset_target)
    AllNoTargetNum = len(listdataset_notarget)

    print('AllTargetNum:',AllTargetNum,' AllNoTargetNum:',AllNoTargetNum)


    random.shuffle(listdataset_target)

    random.shuffle(listdataset_notarget)

    cccc = 0
    count_num = 0
    count_target = 0
    count_notarget = 0
    bc = 0
    while True:

        if count_num<5:
            data_fullpath = listdataset_target[count_target]
            count_target+=1
            # print('target!')
            if count_target>= AllTargetNum:
                count_target = 0
                random.shuffle(listdataset_target)
        elif count_num >= 5:
            data_fullpath = listdataset_notarget[count_notarget]
            count_notarget += 1
            # print('notarget!')
            count_num = 0
            if count_notarget >= AllNoTargetNum:
                count_notarget = 0
                random.shuffle(listdataset_notarget)

        f = h5py.File(data_fullpath)
        # t1_1 = f['t1'][:]
        # t1_1 = np.transpose(t1_1,[1,0,2])
        t2_1 = f['t2'][:]
        t2_1 = np.transpose(t2_1, [1, 0, 2])
        label_1 = f['gt'][:]
        label_1 = np.transpose(label_1, [1, 0, 2])
        f.close()
        # t1_1 = t1_1*1.0
        t2_1 = t2_1*1.0
        # print(np.max(t1_1),np.min(t1_1))
        # print(np.max(t2_1), np.min(t2_1))

        # ## only t1
        # t1_1 = cv2.resize(t1_1, (512, 512), interpolation=cv2.INTER_CUBIC)
        # t1_1 = np.expand_dims(t1_1, axis=3)

        flip_num = random.randint(-1, 2)
        if flip_num != 2:
            # t1_1 = cv2.flip(t1_1,1,dst=None) #1 shuiping, 0 chuizhi, -1 duijiao
            t2_1 = cv2.flip(t2_1, 1, dst=None)  # 1 shuiping, 0 chuizhi, -1 duijiao
            label_1 = cv2.flip(label_1,1,dst=None)
        #     t1_1 = np.expand_dims(t1_1,axis=3)
            t2_1 = np.expand_dims(t2_1, axis=3)
            label_1 = np.expand_dims(label_1, axis=3)
        #
        # resize_num = random.randint(0,2)
        # if resize_num == 1:
        # #     t1_1_r = cv2.resize(t1_1,(192,192),interpolation=cv2.INTER_CUBIC)
        # #     t1_1 = np.zeros_like(t1_1)
        # #     t1_1[32:224,32:224,0] = t1_1_r
        #     t2_1_r = cv2.resize(t2_1, (384, 384), interpolation=cv2.INTER_CUBIC)
        #     t2_1 = np.zeros_like(t2_1)
        #     t2_1[64:448,64:448,0] = t2_1_r
        #     label_1_r = cv2.resize(label_1, (384, 384), interpolation=cv2.INTER_CUBIC)
        #     label_1 = np.zeros_like(label_1)
        #     label_1[64:448, 64:448,0] = label_1_r
        #     label_1[label_1 > 0.2] = 1
        #     label_1[label_1 <= 0.2] = 0
        # elif resize_num == 2:
        #     # t1_1_r = cv2.resize(t1_1, (320, 320), interpolation=cv2.INTER_CUBIC)
        # #     t1_1 = np.zeros_like(t1_1)
        # #     t1_1[:,:,0] = t1_1_r[32:288,32:288]
        #     t2_1_r = cv2.resize(t2_1, (640, 640), interpolation=cv2.INTER_CUBIC)
        #     t2_1 = np.zeros_like(t2_1)
        #     t2_1[:,:,0] = t2_1_r[64:576,64:576]
        #     label_1_r = cv2.resize(label_1, (640, 640), interpolation=cv2.INTER_CUBIC)
        #     label_1 = np.zeros_like(label_1)
        #     label_1[:,:,0] = label_1_r[64:576, 64:576]
        #     label_1[label_1 > 0.2] = 1
        #     label_1[label_1 <= 0.2] = 0

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
        count_num+=1
        #
        ## test picture
        # cccc+=1
        # image_save_path = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/image_show/'
        # t2_s = t2_1[0,:,:,0]
        # t1_s = t1_1[0,:,:,0]
        # label_s = label_1[0,:,:,0]
        # im_label = np.concatenate((t2_s,label_s),axis=1)
        # im_last = im_label*255
        # im_t1 = t1_s*255
        # new_im = Image.fromarray(im_last.astype(np.uint8))
        # new_im_t1 = Image.fromarray(im_t1.astype(np.uint8))
        # new_im.save(image_save_path+str(cccc)+'.jpg')
        # new_im_t1.save(image_save_path + str(cccc) + '_t1.jpg')

        if bc>= batch_size:
            bc = 0
            yield [data_t2],label_train

def generate_segmentation_data_from_file_t2_SNIP(path,batch_size=1):
    DataDirPath = path
    listpath = os.listdir(DataDirPath)

    dataall_num = len(listpath)

    listdataset_target = []
    listdataset_notarget = []
    listdataset_target.extend(listpath)
    for i in range(0,dataall_num):
        Path_datadir = DataDirPath
        listdataset_target[i] = Path_datadir + listpath[i]

    AllDataNum = len(listdataset_target)
    print('AllDataNum: ',AllDataNum)
    pop_flag = 0
    for num in range(AllDataNum)[::-1]:
        data_path_read = copy.deepcopy(listdataset_target[num])
        f = h5py.File(data_path_read)
        label_0 = f['gt'][:]
        f.close()
        label_0[label_0 > 0] = 1
        if np.sum(label_0) == 0:
            listdataset_notarget.append(data_path_read)
            if pop_flag==0:
                listdataset_target.pop(num)
                pop_flag=1
        pop_flag = 0

    AllTargetNum = len(listdataset_target)
    AllNoTargetNum = len(listdataset_notarget)

    print('AllTargetNum:',AllTargetNum,' AllNoTargetNum:',AllNoTargetNum)


    random.shuffle(listdataset_target)

    random.shuffle(listdataset_notarget)

    cccc = 0
    count_num = 0
    count_target = 0
    count_notarget = 0
    bc = 0
    while True:

        if count_num<5:
            data_fullpath = listdataset_target[count_target]
            count_target+=1
            # print('target!')
            if count_target>= AllTargetNum:
                count_target = 0
                random.shuffle(listdataset_target)
        elif count_num >= 5:
            data_fullpath = listdataset_notarget[count_notarget]
            count_notarget += 1
            # print('notarget!')
            count_num = 0
            if count_notarget >= AllNoTargetNum:
                count_notarget = 0
                random.shuffle(listdataset_notarget)

        f = h5py.File(data_fullpath)
        # t1_1 = f['t1'][:]
        # t1_1 = np.transpose(t1_1,[1,0,2])
        t2_1 = f['t2'][:]
        t2_1 = np.transpose(t2_1, [1, 0, 2])
        label_1 = f['gt'][:]
        label_1 = np.transpose(label_1, [1, 0, 2])
        f.close()
        # t1_1 = t1_1*1.0
        t2_1 = t2_1*1.0
        # print(np.max(t1_1),np.min(t1_1))
        # print(np.max(t2_1), np.min(t2_1))

        # ## only t1
        # t1_1 = cv2.resize(t1_1, (512, 512), interpolation=cv2.INTER_CUBIC)
        # t1_1 = np.expand_dims(t1_1, axis=3)

        flip_num = random.randint(-1, 2)
        if flip_num != 2:
            # t1_1 = cv2.flip(t1_1,1,dst=None) #1 shuiping, 0 chuizhi, -1 duijiao
            t2_1 = cv2.flip(t2_1, 1, dst=None)  # 1 shuiping, 0 chuizhi, -1 duijiao
            label_1 = cv2.flip(label_1,1,dst=None)
            # t1_1 = np.expand_dims(t1_1,axis=3)
            t2_1 = np.expand_dims(t2_1, axis=3)
            label_1 = np.expand_dims(label_1, axis=3)
        #
        resize_num = random.randint(0,2)
        label_mask = np.ones_like(label_1)
        if resize_num == 1:
            # t1_1_r = cv2.resize(t1_1,(192,192),interpolation=cv2.INTER_CUBIC)
            # t1_1 = np.zeros_like(t1_1)
            # t1_1[32:224,32:224,0] = t1_1_r
            t2_1_r = cv2.resize(t2_1, (384, 384), interpolation=cv2.INTER_CUBIC)
            t2_1 = np.zeros_like(t2_1)
            t2_1[64:448,64:448,0] = t2_1_r
            label_1_r = cv2.resize(label_1, (384, 384), interpolation=cv2.INTER_CUBIC)
            label_1 = np.zeros_like(label_1)
            label_1[64:448, 64:448,0] = label_1_r
            label_1[label_1 > 0.2] = 1
            label_1[label_1 <= 0.2] = 0
        elif resize_num == 2:
            # t1_1_r = cv2.resize(t1_1, (320, 320), interpolation=cv2.INTER_CUBIC)
            # t1_1 = np.zeros_like(t1_1)
            # t1_1[:,:,0] = t1_1_r[32:288,32:288]
            t2_1_r = cv2.resize(t2_1, (640, 640), interpolation=cv2.INTER_CUBIC)
            t2_1 = np.zeros_like(t2_1)
            t2_1[:,:,0] = t2_1_r[64:576,64:576]
            label_1_r = cv2.resize(label_1, (640, 640), interpolation=cv2.INTER_CUBIC)
            label_1 = np.zeros_like(label_1)
            label_1[:,:,0] = label_1_r[64:576, 64:576]
            label_1[label_1 > 0.2] = 1
            label_1[label_1 <= 0.2] = 0
            label_mask = np.ones_like(label_1)
            label_lty,lty_num = measure.label(label_1,neighbors=8,background=0,return_num=True)
            for lt in range(lty_num):
                true_lty = lt+1
                label_lty_1 = copy.deepcopy(label_lty)
                label_lty_1[label_lty_1!=true_lty] = 0
                label_lty_1[label_lty_1 == true_lty] = 1
                if np.sum(label_lty_1>300):
                    # label_xy = np.where(label_lty_1==1)
                    # x_min = np.min(label_xy[0])-5
                    # y_min = np.min(label_xy[1])-5
                    # x_max = np.max(label_xy[0])+5
                    # y_max = np.max(label_xy[1])+5
                    # label_mask[x_min:x_max,y_min:y_max] = 0
                    label_mask = label_mask * (1-label_lty_1)

        label_1 = np.concatenate([label_1,label_mask],axis=2)
        # print(np.sum(label_1[:,:,0]),np.sum(label_1[:,:,1]))
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
        # count_num+=1

        ## test picture
        # cccc+=1
        # image_save_path = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/image_show/'
        # t2_s = t2_1[0,:,:,0]
        # t1_s = t1_1[0,:,:,0]
        # label_s = label_1[0,:,:,0]
        # im_label = np.concatenate((t2_s,label_s),axis=1)
        # im_last = im_label*255
        # im_t1 = t1_s*255
        # new_im = Image.fromarray(im_last.astype(np.uint8))
        # new_im_t1 = Image.fromarray(im_t1.astype(np.uint8))
        # new_im.save(image_save_path+str(cccc)+'.jpg')
        # new_im_t1.save(image_save_path + str(cccc) + '_t1.jpg')

        if bc>= batch_size:
            bc = 0
            yield [data_t2],label_train

def generate_segmentation_data_from_file_t2_SNIP_valid(path,batch_size=1):
    DataDirPath = path
    listpath = os.listdir(DataDirPath)

    dataall_num = len(listpath)

    listdataset_target = []
    listdataset_notarget = []
    listdataset_target.extend(listpath)
    for i in range(0,dataall_num):
        Path_datadir = DataDirPath
        listdataset_target[i] = Path_datadir + listpath[i]

    AllDataNum = len(listdataset_target)
    print('AllDataNum: ',AllDataNum)
    pop_flag = 0
    for num in range(AllDataNum)[::-1]:
        data_path_read = copy.deepcopy(listdataset_target[num])
        f = h5py.File(data_path_read)
        label_0 = f['gt'][:]
        f.close()
        label_0[label_0 > 0] = 1
        if np.sum(label_0) == 0:
            listdataset_notarget.append(data_path_read)
            if pop_flag==0:
                listdataset_target.pop(num)
                pop_flag=1
        pop_flag = 0

    AllTargetNum = len(listdataset_target)
    AllNoTargetNum = len(listdataset_notarget)

    print('AllTargetNum:',AllTargetNum,' AllNoTargetNum:',AllNoTargetNum)


    random.shuffle(listdataset_target)

    random.shuffle(listdataset_notarget)

    cccc = 0
    count_num = 0
    count_target = 0
    count_notarget = 0
    bc = 0
    while True:

        if count_num<5:
            data_fullpath = listdataset_target[count_target]
            count_target+=1
            # print('target!')
            if count_target>= AllTargetNum:
                count_target = 0
                random.shuffle(listdataset_target)
        elif count_num >= 5:
            data_fullpath = listdataset_notarget[count_notarget]
            count_notarget += 1
            # print('notarget!')
            count_num = 0
            if count_notarget >= AllNoTargetNum:
                count_notarget = 0
                random.shuffle(listdataset_notarget)

        f = h5py.File(data_fullpath)
        # t1_1 = f['t1'][:]
        # t1_1 = np.transpose(t1_1,[1,0,2])
        t2_1 = f['t2'][:]
        t2_1 = np.transpose(t2_1, [1, 0, 2])
        label_1 = f['gt'][:]
        label_1 = np.transpose(label_1, [1, 0, 2])
        f.close()
        # t1_1 = t1_1*1.0
        t2_1 = t2_1*1.0
        # print(np.max(t1_1),np.min(t1_1))
        # print(np.max(t2_1), np.min(t2_1))

        # ## only t1
        # t1_1 = cv2.resize(t1_1, (512, 512), interpolation=cv2.INTER_CUBIC)
        # t1_1 = np.expand_dims(t1_1, axis=3)

        # flip_num = random.randint(-1, 2)
        # if flip_num != 2:
        #     # t1_1 = cv2.flip(t1_1,1,dst=None) #1 shuiping, 0 chuizhi, -1 duijiao
        #     t2_1 = cv2.flip(t2_1, 1, dst=None)  # 1 shuiping, 0 chuizhi, -1 duijiao
        #     label_1 = cv2.flip(label_1,1,dst=None)
        #     # t1_1 = np.expand_dims(t1_1,axis=3)
        #     t2_1 = np.expand_dims(t2_1, axis=3)
        #     label_1 = np.expand_dims(label_1, axis=3)
        #
        # resize_num = random.randint(0,2)
        label_mask = np.ones_like(label_1)
        # if resize_num == 1:
        #     # t1_1_r = cv2.resize(t1_1,(192,192),interpolation=cv2.INTER_CUBIC)
        #     # t1_1 = np.zeros_like(t1_1)
        #     # t1_1[32:224,32:224,0] = t1_1_r
        #     t2_1_r = cv2.resize(t2_1, (384, 384), interpolation=cv2.INTER_CUBIC)
        #     t2_1 = np.zeros_like(t2_1)
        #     t2_1[64:448,64:448,0] = t2_1_r
        #     label_1_r = cv2.resize(label_1, (384, 384), interpolation=cv2.INTER_CUBIC)
        #     label_1 = np.zeros_like(label_1)
        #     label_1[64:448, 64:448,0] = label_1_r
        #     label_1[label_1 > 0.2] = 1
        #     label_1[label_1 <= 0.2] = 0
        # elif resize_num == 2:
        #     # t1_1_r = cv2.resize(t1_1, (320, 320), interpolation=cv2.INTER_CUBIC)
        #     # t1_1 = np.zeros_like(t1_1)
        #     # t1_1[:,:,0] = t1_1_r[32:288,32:288]
        #     t2_1_r = cv2.resize(t2_1, (640, 640), interpolation=cv2.INTER_CUBIC)
        #     t2_1 = np.zeros_like(t2_1)
        #     t2_1[:,:,0] = t2_1_r[64:576,64:576]
        #     label_1_r = cv2.resize(label_1, (640, 640), interpolation=cv2.INTER_CUBIC)
        #     label_1 = np.zeros_like(label_1)
        #     label_1[:,:,0] = label_1_r[64:576, 64:576]
        #     label_1[label_1 > 0.2] = 1
        #     label_1[label_1 <= 0.2] = 0
        #     label_mask = np.ones_like(label_1)
        #     label_lty,lty_num = measure.label(label_1,neighbors=8,background=0,return_num=True)
        #     for lt in range(lty_num):
        #         true_lty = lt+1
        #         label_lty_1 = copy.deepcopy(label_lty)
        #         label_lty_1[label_lty_1!=true_lty] = 0
        #         label_lty_1[label_lty_1 == true_lty] = 1
        #         if np.sum(label_lty_1>300):
        #             # label_xy = np.where(label_lty_1==1)
        #             # x_min = np.min(label_xy[0])-5
        #             # y_min = np.min(label_xy[1])-5
        #             # x_max = np.max(label_xy[0])+5
        #             # y_max = np.max(label_xy[1])+5
        #             # label_mask[x_min:x_max,y_min:y_max] = 0
        #             label_mask = label_mask * (1-label_lty_1)

        label_1 = np.concatenate([label_1,label_mask],axis=2)
        # print(np.sum(label_1[:,:,0]),np.sum(label_1[:,:,1]))
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
        # count_num+=1

        ## test picture
        # cccc+=1
        # image_save_path = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/image_show/'
        # t2_s = t2_1[0,:,:,0]
        # t1_s = t1_1[0,:,:,0]
        # label_s = label_1[0,:,:,0]
        # im_label = np.concatenate((t2_s,label_s),axis=1)
        # im_last = im_label*255
        # im_t1 = t1_s*255
        # new_im = Image.fromarray(im_last.astype(np.uint8))
        # new_im_t1 = Image.fromarray(im_t1.astype(np.uint8))
        # new_im.save(image_save_path+str(cccc)+'.jpg')
        # new_im_t1.save(image_save_path + str(cccc) + '_t1.jpg')

        if bc>= batch_size:
            bc = 0
            yield [data_t2],label_train

def generate_segmentation_data_from_file_SNIP(path,batch_size=1):
    DataDirPath = path
    listpath = os.listdir(DataDirPath)

    dataall_num = len(listpath)

    listdataset_target = []
    listdataset_notarget = []
    listdataset_target.extend(listpath)
    for i in range(0,dataall_num):
        Path_datadir = DataDirPath
        listdataset_target[i] = Path_datadir + listpath[i]

    AllDataNum = len(listdataset_target)
    print('AllDataNum: ',AllDataNum)
    pop_flag = 0
    for num in range(AllDataNum)[::-1]:
        data_path_read = copy.deepcopy(listdataset_target[num])
        f = h5py.File(data_path_read)
        label_0 = f['gt'][:]
        f.close()
        label_0[label_0 > 0] = 1
        if np.sum(label_0) == 0:
            listdataset_notarget.append(data_path_read)
            if pop_flag==0:
                listdataset_target.pop(num)
                pop_flag=1
        pop_flag = 0

    AllTargetNum = len(listdataset_target)
    AllNoTargetNum = len(listdataset_notarget)

    print('AllTargetNum:',AllTargetNum,' AllNoTargetNum:',AllNoTargetNum)


    random.shuffle(listdataset_target)

    random.shuffle(listdataset_notarget)

    cccc = 0
    count_num = 0
    count_target = 0
    count_notarget = 0
    bc = 0
    while True:

        if count_num<5:
            data_fullpath = listdataset_target[count_target]
            count_target+=1
            # print('target!')
            if count_target>= AllTargetNum:
                count_target = 0
                random.shuffle(listdataset_target)
        elif count_num >= 5:
            data_fullpath = listdataset_notarget[count_notarget]
            count_notarget += 1
            # print('notarget!')
            count_num = 0
            if count_notarget >= AllNoTargetNum:
                count_notarget = 0
                random.shuffle(listdataset_notarget)

        f = h5py.File(data_fullpath)
        t1_1 = f['t1'][:]
        t1_1 = np.transpose(t1_1,[1,0,2])
        t2_1 = f['t2'][:]
        t2_1 = np.transpose(t2_1, [1, 0, 2])
        label_1 = f['gt'][:]
        label_1 = np.transpose(label_1, [1, 0, 2])
        f.close()
        t1_1 = t1_1*1.0
        t2_1 = t2_1*1.0
        # print(np.max(t1_1),np.min(t1_1))
        # print(np.max(t2_1), np.min(t2_1))

        # ## only t1
        # t1_1 = cv2.resize(t1_1, (512, 512), interpolation=cv2.INTER_CUBIC)
        # t1_1 = np.expand_dims(t1_1, axis=3)

        flip_num = random.randint(-1, 2)
        if flip_num != 2:
            t1_1 = cv2.flip(t1_1,1,dst=None) #1 shuiping, 0 chuizhi, -1 duijiao
            t2_1 = cv2.flip(t2_1, 1, dst=None)  # 1 shuiping, 0 chuizhi, -1 duijiao
            label_1 = cv2.flip(label_1,1,dst=None)
            t1_1 = np.expand_dims(t1_1,axis=3)
            t2_1 = np.expand_dims(t2_1, axis=3)
            label_1 = np.expand_dims(label_1, axis=3)
        #
        resize_num = random.randint(0,2)
        label_mask = np.ones_like(label_1)
        if resize_num == 1:
            t1_1_r = cv2.resize(t1_1,(192,192),interpolation=cv2.INTER_CUBIC)
            t1_1 = np.zeros_like(t1_1)
            t1_1[32:224,32:224,0] = t1_1_r
            t2_1_r = cv2.resize(t2_1, (384, 384), interpolation=cv2.INTER_CUBIC)
            t2_1 = np.zeros_like(t2_1)
            t2_1[64:448,64:448,0] = t2_1_r
            label_1_r = cv2.resize(label_1, (384, 384), interpolation=cv2.INTER_CUBIC)
            label_1 = np.zeros_like(label_1)
            label_1[64:448, 64:448,0] = label_1_r
            label_1[label_1 > 0.2] = 1
            label_1[label_1 <= 0.2] = 0
        elif resize_num == 2:
            t1_1_r = cv2.resize(t1_1, (320, 320), interpolation=cv2.INTER_CUBIC)
            t1_1 = np.zeros_like(t1_1)
            t1_1[:,:,0] = t1_1_r[32:288,32:288]
            t2_1_r = cv2.resize(t2_1, (640, 640), interpolation=cv2.INTER_CUBIC)
            t2_1 = np.zeros_like(t2_1)
            t2_1[:,:,0] = t2_1_r[64:576,64:576]
            label_1_r = cv2.resize(label_1, (640, 640), interpolation=cv2.INTER_CUBIC)
            label_1 = np.zeros_like(label_1)
            label_1[:,:,0] = label_1_r[64:576, 64:576]
            label_1[label_1 > 0.2] = 1
            label_1[label_1 <= 0.2] = 0
            label_mask = np.ones_like(label_1)
            label_lty,lty_num = measure.label(label_1,neighbors=8,background=0,return_num=True)
            for lt in range(lty_num):
                true_lty = lt+1
                label_lty_1 = copy.deepcopy(label_lty)
                label_lty_1[label_lty_1!=true_lty] = 0
                label_lty_1[label_lty_1 == true_lty] = 1
                if np.sum(label_lty_1>300):
                    # label_xy = np.where(label_lty_1==1)
                    # x_min = np.min(label_xy[0])-5
                    # y_min = np.min(label_xy[1])-5
                    # x_max = np.max(label_xy[0])+5
                    # y_max = np.max(label_xy[1])+5
                    # label_mask[x_min:x_max,y_min:y_max] = 0
                    label_mask = label_mask * (1-label_lty_1)

        label_1 = np.concatenate([label_1,label_mask],axis=2)
        # print(np.sum(label_1[:,:,0]),np.sum(label_1[:,:,1]))
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
        # count_num+=1

        ## test picture
        # cccc+=1
        # image_save_path = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/image_show/'
        # t2_s = t2_1[0,:,:,0]
        # t1_s = t1_1[0,:,:,0]
        # label_s = label_1[0,:,:,0]
        # im_label = np.concatenate((t2_s,label_s),axis=1)
        # im_last = im_label*255
        # im_t1 = t1_s*255
        # new_im = Image.fromarray(im_last.astype(np.uint8))
        # new_im_t1 = Image.fromarray(im_t1.astype(np.uint8))
        # new_im.save(image_save_path+str(cccc)+'.jpg')
        # new_im_t1.save(image_save_path + str(cccc) + '_t1.jpg')

        if bc>= batch_size:
            bc = 0
            yield [data_t2,data_t1],label_train

def generate_segmentation_data_from_file_SNIP_valid(path,batch_size=1):
    DataDirPath = path
    listpath = os.listdir(DataDirPath)

    dataall_num = len(listpath)

    listdataset_target = []
    listdataset_notarget = []
    listdataset_target.extend(listpath)
    for i in range(0,dataall_num):
        Path_datadir = DataDirPath
        listdataset_target[i] = Path_datadir + listpath[i]

    AllDataNum = len(listdataset_target)
    print('AllDataNum: ',AllDataNum)
    pop_flag = 0
    for num in range(AllDataNum)[::-1]:
        data_path_read = copy.deepcopy(listdataset_target[num])
        f = h5py.File(data_path_read)
        label_0 = f['gt'][:]
        f.close()
        label_0[label_0 > 0] = 1
        if np.sum(label_0) == 0:
            listdataset_notarget.append(data_path_read)
            if pop_flag==0:
                listdataset_target.pop(num)
                pop_flag=1
        pop_flag = 0

    AllTargetNum = len(listdataset_target)
    AllNoTargetNum = len(listdataset_notarget)

    print('AllTargetNum:',AllTargetNum,' AllNoTargetNum:',AllNoTargetNum)


    random.shuffle(listdataset_target)

    random.shuffle(listdataset_notarget)

    cccc = 0
    count_num = 0
    count_target = 0
    count_notarget = 0
    bc = 0
    while True:

        if count_num<5:
            data_fullpath = listdataset_target[count_target]
            count_target+=1
            # print('target!')
            if count_target>= AllTargetNum:
                count_target = 0
                random.shuffle(listdataset_target)
        elif count_num >= 5:
            data_fullpath = listdataset_notarget[count_notarget]
            count_notarget += 1
            # print('notarget!')
            count_num = 0
            if count_notarget >= AllNoTargetNum:
                count_notarget = 0
                random.shuffle(listdataset_notarget)

        f = h5py.File(data_fullpath)
        t1_1 = f['t1'][:]
        t1_1 = np.transpose(t1_1,[1,0,2])
        t2_1 = f['t2'][:]
        t2_1 = np.transpose(t2_1, [1, 0, 2])
        label_1 = f['gt'][:]
        label_1 = np.transpose(label_1, [1, 0, 2])
        f.close()
        t1_1 = t1_1*1.0
        t2_1 = t2_1*1.0
        # print(np.max(t1_1),np.min(t1_1))
        # print(np.max(t2_1), np.min(t2_1))

        # ## only t1
        # t1_1 = cv2.resize(t1_1, (512, 512), interpolation=cv2.INTER_CUBIC)
        # t1_1 = np.expand_dims(t1_1, axis=3)

        # flip_num = random.randint(-1, 2)
        # if flip_num != 2:
        #     # t1_1 = cv2.flip(t1_1,1,dst=None) #1 shuiping, 0 chuizhi, -1 duijiao
        #     t2_1 = cv2.flip(t2_1, 1, dst=None)  # 1 shuiping, 0 chuizhi, -1 duijiao
        #     label_1 = cv2.flip(label_1,1,dst=None)
        #     # t1_1 = np.expand_dims(t1_1,axis=3)
        #     t2_1 = np.expand_dims(t2_1, axis=3)
        #     label_1 = np.expand_dims(label_1, axis=3)
        #
        # resize_num = random.randint(0,2)
        label_mask = np.ones_like(label_1)
        # if resize_num == 1:
        #     # t1_1_r = cv2.resize(t1_1,(192,192),interpolation=cv2.INTER_CUBIC)
        #     # t1_1 = np.zeros_like(t1_1)
        #     # t1_1[32:224,32:224,0] = t1_1_r
        #     t2_1_r = cv2.resize(t2_1, (384, 384), interpolation=cv2.INTER_CUBIC)
        #     t2_1 = np.zeros_like(t2_1)
        #     t2_1[64:448,64:448,0] = t2_1_r
        #     label_1_r = cv2.resize(label_1, (384, 384), interpolation=cv2.INTER_CUBIC)
        #     label_1 = np.zeros_like(label_1)
        #     label_1[64:448, 64:448,0] = label_1_r
        #     label_1[label_1 > 0.2] = 1
        #     label_1[label_1 <= 0.2] = 0
        # elif resize_num == 2:
        #     # t1_1_r = cv2.resize(t1_1, (320, 320), interpolation=cv2.INTER_CUBIC)
        #     # t1_1 = np.zeros_like(t1_1)
        #     # t1_1[:,:,0] = t1_1_r[32:288,32:288]
        #     t2_1_r = cv2.resize(t2_1, (640, 640), interpolation=cv2.INTER_CUBIC)
        #     t2_1 = np.zeros_like(t2_1)
        #     t2_1[:,:,0] = t2_1_r[64:576,64:576]
        #     label_1_r = cv2.resize(label_1, (640, 640), interpolation=cv2.INTER_CUBIC)
        #     label_1 = np.zeros_like(label_1)
        #     label_1[:,:,0] = label_1_r[64:576, 64:576]
        #     label_1[label_1 > 0.2] = 1
        #     label_1[label_1 <= 0.2] = 0
        #     label_mask = np.ones_like(label_1)
        #     label_lty,lty_num = measure.label(label_1,neighbors=8,background=0,return_num=True)
        #     for lt in range(lty_num):
        #         true_lty = lt+1
        #         label_lty_1 = copy.deepcopy(label_lty)
        #         label_lty_1[label_lty_1!=true_lty] = 0
        #         label_lty_1[label_lty_1 == true_lty] = 1
        #         if np.sum(label_lty_1>300):
        #             # label_xy = np.where(label_lty_1==1)
        #             # x_min = np.min(label_xy[0])-5
        #             # y_min = np.min(label_xy[1])-5
        #             # x_max = np.max(label_xy[0])+5
        #             # y_max = np.max(label_xy[1])+5
        #             # label_mask[x_min:x_max,y_min:y_max] = 0
        #             label_mask = label_mask * (1-label_lty_1)

        label_1 = np.concatenate([label_1,label_mask],axis=2)
        # print(np.sum(label_1[:,:,0]),np.sum(label_1[:,:,1]))
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
        # count_num+=1

        ## test picture
        # cccc+=1
        # image_save_path = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/image_show/'
        # t2_s = t2_1[0,:,:,0]
        # t1_s = t1_1[0,:,:,0]
        # label_s = label_1[0,:,:,0]
        # im_label = np.concatenate((t2_s,label_s),axis=1)
        # im_last = im_label*255
        # im_t1 = t1_s*255
        # new_im = Image.fromarray(im_last.astype(np.uint8))
        # new_im_t1 = Image.fromarray(im_t1.astype(np.uint8))
        # new_im.save(image_save_path+str(cccc)+'.jpg')
        # new_im_t1.save(image_save_path + str(cccc) + '_t1.jpg')

        if bc>= batch_size:
            bc = 0
            yield [data_t2,data_t1],label_train

def generate_segmentation_data_from_file_t2_SNIP_FPN(path,batch_size=1):
    DataDirPath = path
    listpath = os.listdir(DataDirPath)

    dataall_num = len(listpath)

    listdataset_target = []
    listdataset_notarget = []
    listdataset_target.extend(listpath)
    for i in range(0,dataall_num):
        Path_datadir = DataDirPath
        listdataset_target[i] = Path_datadir + listpath[i]

    AllDataNum = len(listdataset_target)
    print('AllDataNum: ',AllDataNum)
    pop_flag = 0
    for num in range(AllDataNum)[::-1]:
        data_path_read = copy.deepcopy(listdataset_target[num])
        f = h5py.File(data_path_read)
        label_0 = f['gt'][:]
        f.close()
        label_0[label_0 > 0] = 1
        if np.sum(label_0) == 0:
            listdataset_notarget.append(data_path_read)
            if pop_flag==0:
                listdataset_target.pop(num)
                pop_flag=1
        pop_flag = 0

    AllTargetNum = len(listdataset_target)
    AllNoTargetNum = len(listdataset_notarget)

    print('AllTargetNum:',AllTargetNum,' AllNoTargetNum:',AllNoTargetNum)


    random.shuffle(listdataset_target)

    random.shuffle(listdataset_notarget)

    cccc = 0
    count_num = 0
    count_target = 0
    count_notarget = 0
    bc = 0
    while True:

        if count_num<5:
            data_fullpath = listdataset_target[count_target]
            count_target+=1
            # print('target!')
            if count_target>= AllTargetNum:
                count_target = 0
                random.shuffle(listdataset_target)
        elif count_num >= 5:
            data_fullpath = listdataset_notarget[count_notarget]
            count_notarget += 1
            # print('notarget!')
            count_num = 0
            if count_notarget >= AllNoTargetNum:
                count_notarget = 0
                random.shuffle(listdataset_notarget)

        f = h5py.File(data_fullpath)
        # t1_1 = f['t1'][:]
        # t1_1 = np.transpose(t1_1,[1,0,2])
        t2_1 = f['t2'][:]
        t2_1 = np.transpose(t2_1, [1, 0, 2])
        label_1 = f['gt'][:]
        label_1 = np.transpose(label_1, [1, 0, 2])
        f.close()
        # t1_1 = t1_1*1.0
        t2_1 = t2_1*1.0
        # print(np.max(t1_1),np.min(t1_1))
        # print(np.max(t2_1), np.min(t2_1))

        # ## only t1
        # t1_1 = cv2.resize(t1_1, (512, 512), interpolation=cv2.INTER_CUBIC)
        # t1_1 = np.expand_dims(t1_1, axis=3)

        flip_num = random.randint(-1, 2)
        if flip_num != 2:
            # t1_1 = cv2.flip(t1_1,1,dst=None) #1 shuiping, 0 chuizhi, -1 duijiao
            t2_1 = cv2.flip(t2_1, 1, dst=None)  # 1 shuiping, 0 chuizhi, -1 duijiao
            label_1 = cv2.flip(label_1,1,dst=None)
            # t1_1 = np.expand_dims(t1_1,axis=3)
            t2_1 = np.expand_dims(t2_1, axis=3)
            label_1 = np.expand_dims(label_1, axis=3)
        #
        resize_num = random.randint(0,2)
        label_mask = np.ones_like(label_1)
        if resize_num == 1:
            # t1_1_r = cv2.resize(t1_1,(192,192),interpolation=cv2.INTER_CUBIC)
            # t1_1 = np.zeros_like(t1_1)
            # t1_1[32:224,32:224,0] = t1_1_r
            t2_1_r = cv2.resize(t2_1, (384, 384), interpolation=cv2.INTER_CUBIC)
            t2_1 = np.zeros_like(t2_1)
            t2_1[64:448,64:448,0] = t2_1_r
            label_1_r = cv2.resize(label_1, (384, 384), interpolation=cv2.INTER_CUBIC)
            label_1 = np.zeros_like(label_1)
            label_1[64:448, 64:448,0] = label_1_r
            label_1[label_1 > 0.2] = 1
            label_1[label_1 <= 0.2] = 0
        elif resize_num == 2:
            # t1_1_r = cv2.resize(t1_1, (320, 320), interpolation=cv2.INTER_CUBIC)
            # t1_1 = np.zeros_like(t1_1)
            # t1_1[:,:,0] = t1_1_r[32:288,32:288]
            t2_1_r = cv2.resize(t2_1, (640, 640), interpolation=cv2.INTER_CUBIC)
            t2_1 = np.zeros_like(t2_1)
            t2_1[:,:,0] = t2_1_r[64:576,64:576]
            label_1_r = cv2.resize(label_1, (640, 640), interpolation=cv2.INTER_CUBIC)
            label_1 = np.zeros_like(label_1)
            label_1[:,:,0] = label_1_r[64:576, 64:576]
            label_1[label_1 > 0.2] = 1
            label_1[label_1 <= 0.2] = 0
            label_mask = np.ones_like(label_1)
            label_lty,lty_num = measure.label(label_1,neighbors=8,background=0,return_num=True)
            for lt in range(lty_num):
                true_lty = lt+1
                label_lty_1 = copy.deepcopy(label_lty)
                label_lty_1[label_lty_1!=true_lty] = 0
                label_lty_1[label_lty_1 == true_lty] = 1
                if np.sum(label_lty_1>300):
                    # label_xy = np.where(label_lty_1==1)
                    # x_min = np.min(label_xy[0])-5
                    # y_min = np.min(label_xy[1])-5
                    # x_max = np.max(label_xy[0])+5
                    # y_max = np.max(label_xy[1])+5
                    # label_mask[x_min:x_max,y_min:y_max] = 0
                    label_mask = label_mask * (1-label_lty_1)

        label_1 = np.concatenate([label_1,label_mask],axis=2)
        # print(np.sum(label_1[:,:,0]),np.sum(label_1[:,:,1]))
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
        # count_num+=1

        ## test picture
        # cccc+=1
        # image_save_path = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/image_show/'
        # t2_s = t2_1[0,:,:,0]
        # t1_s = t1_1[0,:,:,0]
        # label_s = label_1[0,:,:,0]
        # im_label = np.concatenate((t2_s,label_s),axis=1)
        # im_last = im_label*255
        # im_t1 = t1_s*255
        # new_im = Image.fromarray(im_last.astype(np.uint8))
        # new_im_t1 = Image.fromarray(im_t1.astype(np.uint8))
        # new_im.save(image_save_path+str(cccc)+'.jpg')
        # new_im_t1.save(image_save_path + str(cccc) + '_t1.jpg')

        if bc>= batch_size:
            bc = 0
            yield [data_t2],[label_train,label_train,label_train]

def generate_segmentation_data_from_file_t2_SNIP_FPN_valid(path,batch_size=1):
    DataDirPath = path
    listpath = os.listdir(DataDirPath)

    dataall_num = len(listpath)

    listdataset_target = []
    listdataset_notarget = []
    listdataset_target.extend(listpath)
    for i in range(0,dataall_num):
        Path_datadir = DataDirPath
        listdataset_target[i] = Path_datadir + listpath[i]

    AllDataNum = len(listdataset_target)
    print('AllDataNum: ',AllDataNum)
    pop_flag = 0
    for num in range(AllDataNum)[::-1]:
        data_path_read = copy.deepcopy(listdataset_target[num])
        f = h5py.File(data_path_read)
        label_0 = f['gt'][:]
        f.close()
        label_0[label_0 > 0] = 1
        if np.sum(label_0) == 0:
            listdataset_notarget.append(data_path_read)
            if pop_flag==0:
                listdataset_target.pop(num)
                pop_flag=1
        pop_flag = 0

    AllTargetNum = len(listdataset_target)
    AllNoTargetNum = len(listdataset_notarget)

    print('AllTargetNum:',AllTargetNum,' AllNoTargetNum:',AllNoTargetNum)


    random.shuffle(listdataset_target)

    random.shuffle(listdataset_notarget)

    cccc = 0
    count_num = 0
    count_target = 0
    count_notarget = 0
    bc = 0
    while True:

        if count_num<5:
            data_fullpath = listdataset_target[count_target]
            count_target+=1
            # print('target!')
            if count_target>= AllTargetNum:
                count_target = 0
                random.shuffle(listdataset_target)
        elif count_num >= 5:
            data_fullpath = listdataset_notarget[count_notarget]
            count_notarget += 1
            # print('notarget!')
            count_num = 0
            if count_notarget >= AllNoTargetNum:
                count_notarget = 0
                random.shuffle(listdataset_notarget)

        f = h5py.File(data_fullpath)
        # t1_1 = f['t1'][:]
        # t1_1 = np.transpose(t1_1,[1,0,2])
        t2_1 = f['t2'][:]
        t2_1 = np.transpose(t2_1, [1, 0, 2])
        label_1 = f['gt'][:]
        label_1 = np.transpose(label_1, [1, 0, 2])
        f.close()
        # t1_1 = t1_1*1.0
        t2_1 = t2_1*1.0
        # print(np.max(t1_1),np.min(t1_1))
        # print(np.max(t2_1), np.min(t2_1))

        # ## only t1
        # t1_1 = cv2.resize(t1_1, (512, 512), interpolation=cv2.INTER_CUBIC)
        # t1_1 = np.expand_dims(t1_1, axis=3)

        # flip_num = random.randint(-1, 2)
        # if flip_num != 2:
        #     # t1_1 = cv2.flip(t1_1,1,dst=None) #1 shuiping, 0 chuizhi, -1 duijiao
        #     t2_1 = cv2.flip(t2_1, 1, dst=None)  # 1 shuiping, 0 chuizhi, -1 duijiao
        #     label_1 = cv2.flip(label_1,1,dst=None)
        #     # t1_1 = np.expand_dims(t1_1,axis=3)
        #     t2_1 = np.expand_dims(t2_1, axis=3)
        #     label_1 = np.expand_dims(label_1, axis=3)
        #
        # resize_num = random.randint(0,2)
        label_mask = np.ones_like(label_1)
        # if resize_num == 1:
        #     # t1_1_r = cv2.resize(t1_1,(192,192),interpolation=cv2.INTER_CUBIC)
        #     # t1_1 = np.zeros_like(t1_1)
        #     # t1_1[32:224,32:224,0] = t1_1_r
        #     t2_1_r = cv2.resize(t2_1, (384, 384), interpolation=cv2.INTER_CUBIC)
        #     t2_1 = np.zeros_like(t2_1)
        #     t2_1[64:448,64:448,0] = t2_1_r
        #     label_1_r = cv2.resize(label_1, (384, 384), interpolation=cv2.INTER_CUBIC)
        #     label_1 = np.zeros_like(label_1)
        #     label_1[64:448, 64:448,0] = label_1_r
        #     label_1[label_1 > 0.2] = 1
        #     label_1[label_1 <= 0.2] = 0
        # elif resize_num == 2:
        #     # t1_1_r = cv2.resize(t1_1, (320, 320), interpolation=cv2.INTER_CUBIC)
        #     # t1_1 = np.zeros_like(t1_1)
        #     # t1_1[:,:,0] = t1_1_r[32:288,32:288]
        #     t2_1_r = cv2.resize(t2_1, (640, 640), interpolation=cv2.INTER_CUBIC)
        #     t2_1 = np.zeros_like(t2_1)
        #     t2_1[:,:,0] = t2_1_r[64:576,64:576]
        #     label_1_r = cv2.resize(label_1, (640, 640), interpolation=cv2.INTER_CUBIC)
        #     label_1 = np.zeros_like(label_1)
        #     label_1[:,:,0] = label_1_r[64:576, 64:576]
        #     label_1[label_1 > 0.2] = 1
        #     label_1[label_1 <= 0.2] = 0
        #     label_mask = np.ones_like(label_1)
        #     label_lty,lty_num = measure.label(label_1,neighbors=8,background=0,return_num=True)
        #     for lt in range(lty_num):
        #         true_lty = lt+1
        #         label_lty_1 = copy.deepcopy(label_lty)
        #         label_lty_1[label_lty_1!=true_lty] = 0
        #         label_lty_1[label_lty_1 == true_lty] = 1
        #         if np.sum(label_lty_1>300):
        #             # label_xy = np.where(label_lty_1==1)
        #             # x_min = np.min(label_xy[0])-5
        #             # y_min = np.min(label_xy[1])-5
        #             # x_max = np.max(label_xy[0])+5
        #             # y_max = np.max(label_xy[1])+5
        #             # label_mask[x_min:x_max,y_min:y_max] = 0
        #             label_mask = label_mask * (1-label_lty_1)

        label_1 = np.concatenate([label_1,label_mask],axis=2)
        # print(np.sum(label_1[:,:,0]),np.sum(label_1[:,:,1]))
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
        # count_num+=1

        ## test picture
        # cccc+=1
        # image_save_path = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/image_show/'
        # t2_s = t2_1[0,:,:,0]
        # t1_s = t1_1[0,:,:,0]
        # label_s = label_1[0,:,:,0]
        # im_label = np.concatenate((t2_s,label_s),axis=1)
        # im_last = im_label*255
        # im_t1 = t1_s*255
        # new_im = Image.fromarray(im_last.astype(np.uint8))
        # new_im_t1 = Image.fromarray(im_t1.astype(np.uint8))
        # new_im.save(image_save_path+str(cccc)+'.jpg')
        # new_im_t1.save(image_save_path + str(cccc) + '_t1.jpg')

        if bc>= batch_size:
            bc = 0
            yield [data_t2],[label_train,label_train,label_train]

def generate_segmentation_data_from_file_t1(path,batch_size=1):
    DataDirPath = path
    listpath = os.listdir(DataDirPath)

    dataall_num = len(listpath)

    listdataset_target = []
    listdataset_notarget = []
    listdataset_target.extend(listpath)
    for i in range(0,dataall_num):
        Path_datadir = DataDirPath
        listdataset_target[i] = Path_datadir + listpath[i]

    AllDataNum = len(listdataset_target)
    print('AllDataNum: ',AllDataNum)
    pop_flag = 0
    for num in range(AllDataNum)[::-1]:
        data_path_read = copy.deepcopy(listdataset_target[num])
        f = h5py.File(data_path_read)
        label_0 = f['gt'][:]
        f.close()
        label_0[label_0 > 0] = 1
        if np.sum(label_0) == 0:
            listdataset_notarget.append(data_path_read)
            if pop_flag==0:
                listdataset_target.pop(num)
                pop_flag=1
        pop_flag = 0

    AllTargetNum = len(listdataset_target)
    AllNoTargetNum = len(listdataset_notarget)

    print('AllTargetNum:',AllTargetNum,' AllNoTargetNum:',AllNoTargetNum)


    random.shuffle(listdataset_target)

    random.shuffle(listdataset_notarget)

    cccc = 0
    count_num = 0
    count_target = 0
    count_notarget = 0
    bc = 0
    while True:

        if count_num<5:
            data_fullpath = listdataset_target[count_target]
            count_target+=1
            # print('target!')
            if count_target>= AllTargetNum:
                count_target = 0
                random.shuffle(listdataset_target)
        elif count_num >= 5:
            data_fullpath = listdataset_notarget[count_notarget]
            count_notarget += 1
            # print('notarget!')
            count_num = 0
            if count_notarget >= AllNoTargetNum:
                count_notarget = 0
                random.shuffle(listdataset_notarget)

        f = h5py.File(data_fullpath)
        t1_1 = f['t1'][:]
        t1_1 = np.transpose(t1_1,[1,0,2])
        t2_1 = f['t2'][:]
        t2_1 = np.transpose(t2_1, [1, 0, 2])
        label_1 = f['gt'][:]
        label_1 = np.transpose(label_1, [1, 0, 2])
        f.close()
        t1_1 = t1_1*1.0
        t2_1 = t2_1*1.0
        # print(np.max(t1_1),np.min(t1_1))
        # print(np.max(t2_1), np.min(t2_1))

        ## only t1
        t1_1 = cv2.resize(t1_1, (512, 512), interpolation=cv2.INTER_CUBIC)
        t1_1 = np.expand_dims(t1_1, axis=3)

        # flip_num = random.randint(0, 1)
        # if flip_num == 1:
        #     t1_1 = cv2.flip(t1_1,1,dst=None) #1 shuiping, 0 chuizhi, -1 duijiao
        #     t2_1 = cv2.flip(t2_1, 1, dst=None)  # 1 shuiping, 0 chuizhi, -1 duijiao
        #     label_1 = cv2.flip(label_1,1,dst=None)
        #     t1_1 = np.expand_dims(t1_1,axis=3)
        #     t2_1 = np.expand_dims(t2_1, axis=3)
        #     label_1 = np.expand_dims(label_1, axis=3)
        #
        # resize_num = random.randint(0,2)
        # if resize_num == 1:
        #     t1_1_r = cv2.resize(t1_1,(192,192),interpolation=cv2.INTER_CUBIC)
        #     t1_1 = np.zeros_like(t1_1)
        #     t1_1[32:224,32:224,0] = t1_1_r
        #     t2_1_r = cv2.resize(t2_1, (384, 384), interpolation=cv2.INTER_CUBIC)
        #     t2_1 = np.zeros_like(t2_1)
        #     t2_1[64:448,64:448,0] = t2_1_r
        #     label_1_r = cv2.resize(label_1, (384, 384), interpolation=cv2.INTER_CUBIC)
        #     label_1 = np.zeros_like(label_1)
        #     label_1[64:448, 64:448,0] = label_1_r
        #     label_1[label_1 > 0.2] = 1
        #     label_1[label_1 <= 0.2] = 0
        # elif resize_num == 2:
        #     t1_1_r = cv2.resize(t1_1, (320, 320), interpolation=cv2.INTER_CUBIC)
        #     t1_1 = np.zeros_like(t1_1)
        #     t1_1[:,:,0] = t1_1_r[32:288,32:288]
        #     t2_1_r = cv2.resize(t2_1, (640, 640), interpolation=cv2.INTER_CUBIC)
        #     t2_1 = np.zeros_like(t2_1)
        #     t2_1[:,:,0] = t2_1_r[64:576,64:576]
        #     label_1_r = cv2.resize(label_1, (640, 640), interpolation=cv2.INTER_CUBIC)
        #     label_1 = np.zeros_like(label_1)
        #     label_1[:,:,0] = label_1_r[64:576, 64:576]
        #     label_1[label_1 > 0.2] = 1
        #     label_1[label_1 <= 0.2] = 0

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
        count_num+=1

        ## test picture
        # cccc+=1
        # image_save_path = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/image_show/'
        # t2_s = t2_1[0,:,:,0]
        # t1_s = t1_1[0,:,:,0]
        # label_s = label_1[0,:,:,0]
        # im_label = np.concatenate((t2_s,label_s),axis=1)
        # im_last = im_label*255
        # im_t1 = t1_s*255
        # new_im = Image.fromarray(im_last.astype(np.uint8))
        # new_im_t1 = Image.fromarray(im_t1.astype(np.uint8))
        # new_im.save(image_save_path+str(cccc)+'.jpg')
        # new_im_t1.save(image_save_path + str(cccc) + '_t1.jpg')

        if bc>= batch_size:
            bc = 0
            yield [data_t1],label_train

def generate_segmentation_data_from_file_valid(path,batch_size=1):
    DataDirPath = path
    listpath = os.listdir(DataDirPath)

    dataall_num = len(listpath)

    listdataset_target = []
    listdataset_notarget = []
    listdataset_target.extend(listpath)
    for i in range(0,dataall_num):
        Path_datadir = DataDirPath
        listdataset_target[i] = Path_datadir + listpath[i]

    AllDataNum = len(listdataset_target)
    print('AllDataNum: ', AllDataNum)
    pop_flag = 0
    for num in range(AllDataNum)[::-1]:
        data_path_read = copy.deepcopy(listdataset_target[num])
        f = h5py.File(data_path_read)
        label_0 = f['gt'][:]
        f.close()
        label_0[label_0 > 0] = 1
        if np.sum(label_0) == 0:
            listdataset_notarget.append(data_path_read)
            if pop_flag == 0:
                listdataset_target.pop(num)
                pop_flag = 1
        pop_flag = 0

    AllTargetNum = len(listdataset_target)
    AllNoTargetNum = len(listdataset_notarget)

    print('AllTargetNum:', AllTargetNum, ' AllNoTargetNum:', AllNoTargetNum)
    num = 0
    count_target = 0
    count_notarget = 0
    bc = 0
    while True:

        if num < AllTargetNum:
            data_fullpath = listdataset_target[count_target]
            count_target += 1
            if count_target >= AllTargetNum:
                count_target = 0
        else:
            data_fullpath = listdataset_notarget[count_notarget]
            count_notarget += 1
            if count_notarget >= AllNoTargetNum:
                count_notarget = 0

        f = h5py.File(data_fullpath)
        t1_1 = f['t1'][:]
        t1_1 = np.transpose(t1_1,[1,0,2])
        t2_1 = f['t2'][:]
        t2_1 = np.transpose(t2_1, [1, 0, 2])
        label_1 = f['gt'][:]
        label_1 = np.transpose(label_1, [1, 0, 2])
        f.close()

        t1_1 = t1_1*1.0
        t2_1 = t2_1*1.0

        # ## only t1
        # t1_1 = cv2.resize(t1_1, (512, 512), interpolation=cv2.INTER_CUBIC)
        # t1_1 = np.expand_dims(t1_1, axis=3)

        # flip_num = random.randint(-1, 2)
        # if flip_num != 2:
        #     data_1 = cv2.flip(data_1,flip_num,dst=None) #1 shuiping, 0 chuizhi, -1 duijiao
        #     label_1 = cv2.flip(label_1,flip_num,dst=None)

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
        num += 1
        if num>= AllDataNum:
            num = 0
        if bc >= batch_size:
            bc = 0
            yield [data_t2,data_t1], label_train

def generate_segmentation_data_from_file_valid_t2(path,batch_size=1):
    DataDirPath = path
    listpath = os.listdir(DataDirPath)

    dataall_num = len(listpath)

    listdataset_target = []
    listdataset_notarget = []
    listdataset_target.extend(listpath)
    for i in range(0,dataall_num):
        Path_datadir = DataDirPath
        listdataset_target[i] = Path_datadir + listpath[i]

    AllDataNum = len(listdataset_target)
    print('AllDataNum: ', AllDataNum)
    pop_flag = 0
    for num in range(AllDataNum)[::-1]:
        data_path_read = copy.deepcopy(listdataset_target[num])
        f = h5py.File(data_path_read)
        label_0 = f['gt'][:]
        f.close()
        label_0[label_0 > 0] = 1
        if np.sum(label_0) == 0:
            listdataset_notarget.append(data_path_read)
            if pop_flag == 0:
                listdataset_target.pop(num)
                pop_flag = 1
        pop_flag = 0

    AllTargetNum = len(listdataset_target)
    AllNoTargetNum = len(listdataset_notarget)

    print('AllTargetNum:', AllTargetNum, ' AllNoTargetNum:', AllNoTargetNum)
    num = 0
    count_target = 0
    count_notarget = 0
    bc = 0
    while True:

        if num < AllTargetNum:
            data_fullpath = listdataset_target[count_target]
            count_target += 1
            if count_target >= AllTargetNum:
                count_target = 0
        # else:
        #     data_fullpath = listdataset_notarget[count_notarget]
        #     count_notarget += 1
        #     if count_notarget >= AllNoTargetNum:
        #         count_notarget = 0

        f = h5py.File(data_fullpath)
        # t1_1 = f['t1'][:]
        # t1_1 = np.transpose(t1_1,[1,0,2])
        t2_1 = f['t2'][:]
        t2_1 = np.transpose(t2_1, [1, 0, 2])
        label_1 = f['gt'][:]
        label_1 = np.transpose(label_1, [1, 0, 2])
        f.close()

        # t1_1 = t1_1*1.0
        t2_1 = t2_1*1.0

        ## only t1
        # t1_1 = cv2.resize(t1_1, (512, 512), interpolation=cv2.INTER_CUBIC)
        # t1_1 = np.expand_dims(t1_1, axis=3)

        # flip_num = random.randint(-1, 2)
        # if flip_num != 2:
        #     data_1 = cv2.flip(data_1,flip_num,dst=None) #1 shuiping, 0 chuizhi, -1 duijiao
        #     label_1 = cv2.flip(label_1,flip_num,dst=None)

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
        num += 1
        if num>= AllDataNum:
            num = 0
        if bc >= batch_size:
            bc = 0
            yield [data_t2], label_train

def generate_segmentation_data_from_file_test_t2(path,batch_size=1):
    DataDirPath = path
    listpath = os.listdir(DataDirPath)

    dataall_num = len(listpath)

    listdataset_target = []
    listdataset_notarget = []
    listdataset_target.extend(listpath)
    for i in range(0,dataall_num):
        Path_datadir = DataDirPath
        listdataset_target[i] = Path_datadir + listpath[i]

    AllDataNum = len(listdataset_target)
    print('AllDataNum: ', AllDataNum)
    pop_flag = 0
    for num in range(AllDataNum)[::-1]:
        data_path_read = copy.deepcopy(listdataset_target[num])
        f = h5py.File(data_path_read)
        label_0 = f['gt'][:]
        f.close()
        label_0[label_0 > 0] = 1
        if np.sum(label_0) == 0:
            listdataset_notarget.append(data_path_read)
            if pop_flag == 0:
                listdataset_target.pop(num)
                pop_flag = 1
        pop_flag = 0

    AllTargetNum = len(listdataset_target)
    AllNoTargetNum = len(listdataset_notarget)

    print('AllTargetNum:', AllTargetNum, ' AllNoTargetNum:', AllNoTargetNum)
    num = 0
    count_target = 0
    count_notarget = 0
    bc = 0
    while True:

        if num < AllTargetNum:
            data_fullpath = listdataset_target[count_target]
            count_target += 1
            if count_target >= AllTargetNum:
                count_target = 0
        else:
            data_fullpath = listdataset_notarget[count_notarget]
            count_notarget += 1
            if count_notarget >= AllNoTargetNum:
                count_notarget = 0

        f = h5py.File(data_fullpath)
        # t1_1 = f['t1'][:]
        # t1_1 = np.transpose(t1_1,[1,0,2])
        t2_1 = f['t2'][:]
        t2_1 = np.transpose(t2_1, [1, 0, 2])
        label_1 = f['gt'][:]
        label_1 = np.transpose(label_1, [1, 0, 2])
        f.close()

        # t1_1 = t1_1*1.0
        t2_1 = t2_1*1.0

        ## only t1
        # t1_1 = cv2.resize(t1_1, (512, 512), interpolation=cv2.INTER_CUBIC)
        # t1_1 = np.expand_dims(t1_1, axis=3)

        # flip_num = random.randint(-1, 2)
        # if flip_num != 2:
        #     data_1 = cv2.flip(data_1,flip_num,dst=None) #1 shuiping, 0 chuizhi, -1 duijiao
        #     label_1 = cv2.flip(label_1,flip_num,dst=None)

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
        num += 1
        if num>= AllDataNum:
            num = 0
        if bc >= batch_size:
            bc = 0
            yield [data_t2], label_train

def generate_segmentation_data_from_file_valid_t1(path,batch_size=1):
    DataDirPath = path
    listpath = os.listdir(DataDirPath)

    dataall_num = len(listpath)

    listdataset_target = []
    listdataset_notarget = []
    listdataset_target.extend(listpath)
    for i in range(0,dataall_num):
        Path_datadir = DataDirPath
        listdataset_target[i] = Path_datadir + listpath[i]

    AllDataNum = len(listdataset_target)
    print('AllDataNum: ', AllDataNum)
    pop_flag = 0
    for num in range(AllDataNum)[::-1]:
        data_path_read = copy.deepcopy(listdataset_target[num])
        f = h5py.File(data_path_read)
        label_0 = f['gt'][:]
        f.close()
        label_0[label_0 > 0] = 1
        if np.sum(label_0) == 0:
            listdataset_notarget.append(data_path_read)
            if pop_flag == 0:
                listdataset_target.pop(num)
                pop_flag = 1
        pop_flag = 0

    AllTargetNum = len(listdataset_target)
    AllNoTargetNum = len(listdataset_notarget)

    print('AllTargetNum:', AllTargetNum, ' AllNoTargetNum:', AllNoTargetNum)
    num = 0
    count_target = 0
    count_notarget = 0
    bc = 0
    while True:

        if num < AllTargetNum:
            data_fullpath = listdataset_target[count_target]
            count_target += 1
            if count_target >= AllTargetNum:
                count_target = 0
        else:
            data_fullpath = listdataset_notarget[count_notarget]
            count_notarget += 1
            if count_notarget >= AllNoTargetNum:
                count_notarget = 0

        f = h5py.File(data_fullpath)
        t1_1 = f['t1'][:]
        t1_1 = np.transpose(t1_1,[1,0,2])
        t2_1 = f['t2'][:]
        t2_1 = np.transpose(t2_1, [1, 0, 2])
        label_1 = f['gt'][:]
        label_1 = np.transpose(label_1, [1, 0, 2])
        f.close()

        t1_1 = t1_1*1.0
        t2_1 = t2_1*1.0

        ## only t1
        t1_1 = cv2.resize(t1_1, (512, 512), interpolation=cv2.INTER_CUBIC)
        t1_1 = np.expand_dims(t1_1, axis=3)

        # flip_num = random.randint(-1, 2)
        # if flip_num != 2:
        #     data_1 = cv2.flip(data_1,flip_num,dst=None) #1 shuiping, 0 chuizhi, -1 duijiao
        #     label_1 = cv2.flip(label_1,flip_num,dst=None)

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
        num += 1
        if num>= AllDataNum:
            num = 0
        if bc >= batch_size:
            bc = 0
            yield [data_t1], label_train

def generate_classification_data_from_file(path,batch_size=1):
    DataDirPath = path+'original_images/'
    LabelDirPath = path + 'label_images/'
    listpath = os.listdir(DataDirPath)
    # data_train = []
    # label_test = []
    # random.shuffle(listdata)
    dataall_num = len(listpath)

    listdataset_target = []
    listlabelset_target = []
    listdataset_notarget = []
    listlabelset_notarget = []
    listdataset_rea = []
    listlabelset_rea = []
    listdataset_srf = []
    listlabelset_srf = []
    listdataset_ped = []
    listlabelset_ped = []
    for i in range(0,dataall_num):
        data_name = listpath[i] + '/'
        label_name = data_name[:-5]+'_labelMark/'
        Path_datadir = DataDirPath+data_name
        Path_labeldir = LabelDirPath + label_name
        listdata_i = os.listdir(Path_datadir)
        listlabel_i = copy.deepcopy(listdata_i)
        num_i = len(listdata_i)
        for ii in range(num_i):
            listlabel_i[ii] = Path_labeldir + listlabel_i[ii]
            listdata_i[ii] = Path_datadir + listdata_i[ii]
        listdataset_target.extend(listdata_i)
        listlabelset_target.extend(listlabel_i)

    AllDataNum = len(listdataset_target)
    print('AllDataNum: ',AllDataNum)
    pop_flag = 0
    for num in range(AllDataNum)[::-1]:
        label_path_read = copy.deepcopy(listlabelset_target[num])
        data_path_read = copy.deepcopy(listdataset_target[num])
        label_0 = cv2.imread(label_path_read)
        label_0[:, :, 0][label_0[:, :, 0] != 255] = 0
        label_0[:, :, 1][label_0[:, :, 1] != 191] = 0
        label_0[:, :, 2][label_0[:, :, 2] != 128] = 0
        label_0[label_0 > 0] = 1
        if np.sum(label_0) == 0:
            listdataset_notarget.append(data_path_read)
            listlabelset_notarget.append(label_path_read)
            if pop_flag==0:
                listdataset_target.pop(num)
                listlabelset_target.pop(num)
                pop_flag=1
        if np.sum(label_0[:,:,0]) != 0:
            listdataset_rea.append(data_path_read)
            listlabelset_rea.append(label_path_read)
            if pop_flag == 0:
                listdataset_target.pop(num)
                listlabelset_target.pop(num)
                pop_flag = 1
        if np.sum(label_0[:,:,1]) != 0:
            listdataset_srf.append(data_path_read)
            listlabelset_srf.append(label_path_read)
            if pop_flag == 0:
                listdataset_target.pop(num)
                listlabelset_target.pop(num)
                pop_flag = 1
        if np.sum(label_0[:,:,2]) != 0:
            listdataset_ped.append(data_path_read)
            listlabelset_ped.append(label_path_read)
            if pop_flag == 0:
                listdataset_target.pop(num)
                listlabelset_target.pop(num)
                pop_flag = 1

    AllREANum = len(listlabelset_rea)
    AllSRFNum = len(listlabelset_srf)
    AllPEDNum = len(listlabelset_ped)
    AllNotargetNum = len(listlabelset_notarget)
    print('AllREANum:',AllREANum,' AllSRFNum:',AllSRFNum,' AllPEDNum:',AllPEDNum,' AllNotargetNum:',AllNotargetNum)

    shufList = list(zip(listdataset_notarget,listlabelset_notarget))
    random.shuffle(shufList)
    listdataset_notarget[:], listlabelset_notarget[:] = zip(*shufList)

    shufList = list(zip(listdataset_rea,listlabelset_rea))
    random.shuffle(shufList)
    listdataset_rea[:], listlabelset_rea[:] = zip(*shufList)

    shufList = list(zip(listdataset_srf,listlabelset_srf))
    random.shuffle(shufList)
    listdataset_srf[:], listlabelset_srf[:] = zip(*shufList)

    shufList = list(zip(listdataset_ped,listlabelset_ped))
    random.shuffle(shufList)
    listdataset_ped[:], listlabelset_ped[:] = zip(*shufList)

    count_target = 0
    count_rea = 0
    count_srf = 0
    count_ped = 0
    count_notarget = 0
    bc = 0
    while True:

        if count_target==0:
            data_fullpath = listdataset_notarget[count_notarget]
            label_fullpath = listlabelset_notarget[count_notarget]
            count_notarget+=1
            # print('Notarget!')
            if count_notarget>= AllNotargetNum:
                count_notarget = 0
                shufList = list(zip(listdataset_notarget, listlabelset_notarget))
                random.shuffle(shufList)
                listdataset_notarget[:], listlabelset_notarget[:] = zip(*shufList)
        elif count_target==1:
            data_fullpath = listdataset_rea[count_rea]
            label_fullpath = listlabelset_rea[count_rea]
            count_rea+=1
            # print('REA!')
            if count_rea>= AllREANum:
                count_rea = 0
                shufList = list(zip(listdataset_rea, listlabelset_rea))
                random.shuffle(shufList)
                listdataset_rea[:], listlabelset_rea[:] = zip(*shufList)
        elif count_target == 2:
            data_fullpath = listdataset_srf[count_srf]
            label_fullpath = listlabelset_srf[count_srf]
            count_srf += 1
            # print('SRF!')
            if count_srf >= AllSRFNum:
                count_srf = 0
                shufList = list(zip(listdataset_srf, listlabelset_srf))
                random.shuffle(shufList)
                listdataset_srf[:], listlabelset_srf[:] = zip(*shufList)
        elif count_target == 3:
            data_fullpath = listdataset_ped[count_ped]
            label_fullpath = listlabelset_ped[count_ped]
            count_ped += 1
            # print('PED!')
            count_target = -1
            if count_ped >= AllPEDNum:
                count_ped = 0
                shufList = list(zip(listdataset_ped, listlabelset_ped))
                random.shuffle(shufList)
                listdataset_ped[:], listlabelset_ped[:] = zip(*shufList)
        #
        # print(data_fullpath)
        # print(label_fullpath)
        data_1 = cv2.imread(data_fullpath)
        label_1 = cv2.imread(label_fullpath)

        # print('\n',listdata[count % dataset_num])
        data_1 = data_1.astype(float)
        data_1 = np.transpose(data_1,[1,0,2])
        label_1 = label_1.astype(float)
        label_1 = np.transpose(label_1, [1, 0, 2])

        label_class_rea = np.zeros([1, 2], dtype='float')
        label_class_srf = np.zeros([1, 2], dtype='float')
        label_class_ped = np.zeros([1, 2], dtype='float')

        label_1[:, :, 0][label_1[:, :, 0] != 255] = 0
        label_1[:, :, 1][label_1[:, :, 1] != 191] = 0
        label_1[:, :, 2][label_1[:, :, 2] != 128] = 0

        label_1[label_1 > 0] = 1

        if np.sum(label_1[:, :, 0]) != 0:
            label_class_rea[0, 1] = 1
        else:
            label_class_rea[0, 0] = 1

        if np.sum(label_1[:, :, 1]) != 0:
            label_class_srf[0, 1] = 1
        else:
            label_class_srf[0, 0] = 1

        if np.sum(label_1[:, :, 2]) != 0:
            label_class_ped[0, 1] = 1
        else:
            label_class_ped[0, 0] = 1



        data_1 = data_1/255.0

        flip_num = random.randint(-1, 2)
        if flip_num != 2:
            data_1 = cv2.flip(data_1,flip_num,dst=None) #1 shuiping, 0 chuizhi, -1 duijiao

        data_1 = np.expand_dims(data_1, axis=0)
        # label_1 = np.expand_dims(label_1, axis=0)

        if bc == 0:
            data_train = copy.deepcopy(data_1)
            # label_test = label_1
            label_test_cls_rea = copy.deepcopy(label_class_rea)
            label_test_cls_srf = copy.deepcopy(label_class_srf)
            label_test_cls_ped = copy.deepcopy(label_class_ped)
        elif bc > 0:
            data_train = np.concatenate([data_train,data_1],axis=0)
            # label_test = np.concatenate([label_test, label_1], axis=0)
            label_test_cls_rea = np.concatenate([label_test_cls_rea, label_class_rea], axis=0)
            label_test_cls_srf = np.concatenate([label_test_cls_srf, label_class_srf], axis=0)
            label_test_cls_ped = np.concatenate([label_test_cls_ped, label_class_ped], axis=0)
        bc+=1
        count_target+=1

        if bc>= batch_size:
            bc = 0
            yield data_train,[label_test_cls_rea,label_test_cls_srf,label_test_cls_ped]

def generate_classification_data_from_file_valid(path,batch_size=1):
    DataDirPath = path+'original_images/'
    LabelDirPath = path + 'label_images/'
    listpath = os.listdir(DataDirPath)
    # data_train = []
    # label_test = []
    # random.shuffle(listdata)
    dataall_num = len(listpath)

    listdataset_target = []
    listlabelset_target = []
    listdataset_notarget = []
    listlabelset_notarget = []
    for i in range(0,dataall_num):
        data_name = listpath[i] + '/'
        label_name = data_name[:-5]+'_labelMark/'
        Path_datadir = DataDirPath+data_name
        Path_labeldir = LabelDirPath + label_name
        listdata_i = os.listdir(Path_datadir)
        listlabel_i = copy.deepcopy(listdata_i)
        num_i = len(listdata_i)
        for ii in range(num_i):
            listlabel_i[ii] = Path_labeldir + listlabel_i[ii]
            listdata_i[ii] = Path_datadir + listdata_i[ii]
        listdataset_target.extend(listdata_i)
        listlabelset_target.extend(listlabel_i)

    AllDataNum = len(listdataset_target)
    print('\n','AllDataNum: ',AllDataNum)


    count_target = 0

    bc = 0
    while True:
        data_fullpath = listdataset_target[count_target]
        label_fullpath = listlabelset_target[count_target]
        count_target+=1
        # if count_target >= AllDataNum:
        #     continue
        # print(data_fullpath)
        # print(label_fullpath)

        data_1 = cv2.imread(data_fullpath)
        label_1 = cv2.imread(label_fullpath)

        # print('\n',listdata[count % dataset_num])
        data_1 = data_1.astype(float)
        data_1 = np.transpose(data_1,[1,0,2])
        label_1 = label_1.astype(float)
        label_1 = np.transpose(label_1, [1, 0, 2])

        label_class_rea = np.zeros([1,2],dtype='float')
        label_class_srf = np.zeros([1, 2], dtype='float')
        label_class_ped = np.zeros([1, 2], dtype='float')

        label_1[:,:,0][label_1[:,:,0]!=255] = 0
        label_1[:, :, 1][label_1[:, :, 1] != 191] = 0
        label_1[:, :, 2][label_1[:, :, 2] != 128] = 0

        label_1[label_1 > 0] = 1

        if np.sum(label_1[:,:,0])!=0:
            label_class_rea[0,1] = 1
        else:
            label_class_rea[0,0] = 1

        if np.sum(label_1[:,:,1])!=0:
            label_class_srf[0,1] = 1
        else:
            label_class_srf[0,0] = 1

        if np.sum(label_1[:,:,2])!=0:
            label_class_ped[0,1] = 1
        else:
            label_class_ped[0,0] = 1

        data_1 = data_1/255

        data_1 = np.expand_dims(data_1, axis=0)
        # label_1 = np.expand_dims(label_1, axis=0)

        if bc == 0:
            data_train = copy.deepcopy(data_1)
            # label_test = label_1
            label_test_cls_rea = copy.deepcopy(label_class_rea)
            label_test_cls_srf = copy.deepcopy(label_class_srf)
            label_test_cls_ped = copy.deepcopy(label_class_ped)
        elif bc > 0:
            data_train = np.concatenate([data_train,data_1],axis=0)
            # label_test = np.concatenate([label_test, label_1], axis=0)
            label_test_cls_rea = np.concatenate([label_test_cls_rea, label_class_rea], axis=0)
            label_test_cls_srf = np.concatenate([label_test_cls_srf, label_class_srf], axis=0)
            label_test_cls_ped = np.concatenate([label_test_cls_ped, label_class_ped], axis=0)

        bc+=1

        if bc>= batch_size:
            bc = 0
            print(count_target,bc)
            yield data_train,[label_test_cls_rea,label_test_cls_srf,label_test_cls_ped]
            # print(count_target,bc)