import h5py
import numpy as np
import os
import shutil
import random
ori_path = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/data/patient_2D_predict_dsc_test/test/'
target_path = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/data/patient_2D_predict_dsc_test/test_empty/'
data_list = os.listdir(ori_path)
data_num = len(data_list)

for num in range(0,data_num):
    print(num,'/',data_num-1)
    filename = data_list[num]
    filepath = ori_path+filename

    f = h5py.File(filepath)
    label = f['gt'][:]
    f.close()

    if np.sum(label) == 0:
        print(filename)
        newpath = target_path+filename
        shutil.move(filepath,target_path)