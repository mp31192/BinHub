import h5py
import numpy as np
import os

path = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/data/train_1_old_gan_augmentation/'
data_list = os.listdir(path)
data_num = len(data_list)

for num in range(1000):
    print(num)
    data_name = data_list[num]
    data_fullname = path+data_name

    f = h5py.File(data_fullname)
    t2 = f['t2'][:]
    gt = f['gt'][:]
    f.close()

    t2 = np.transpose(t2,[0,2,1,3])
    gt = np.transpose(gt, [0,2,1,3])

    f = h5py.File(data_fullname,'w')
    f['t2'] = t2[0,:,:,:]
    f['gt'] = gt[0,:,:,:]
    f.close()
