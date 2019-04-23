import h5py
import numpy as np
import os
import shutil
import random
train_path = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/data/patient_2D_predict_dsc_train/train_A/'
evalidation_path = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/data/patient_2D_predict_dsc_train/train_B/'
data_list = os.listdir(train_path)
patient_list = []
for dn in data_list:
    dataname = dn[:-3]
    data_str = dataname.split('_')
    patient_num = data_str[1]
    if patient_num not in patient_list:
        patient_list.append(patient_num)



patient_num = len(patient_list)
random.shuffle(patient_list)

tv_ratio = 0.5
knife = int(patient_num*tv_ratio)
num_count = 0
for num in patient_list:
    print("patient num:",num)
    num_count += 1
    if num_count>knife:
        continue
    for i in range(1,51):
        filename = str(i)+'_'+num+'.h5'
        fullpath = train_path+filename
        if os.path.exists(fullpath) == 1:
            print(filename)
            newpath = evalidation_path+filename
            shutil.move(fullpath,newpath)