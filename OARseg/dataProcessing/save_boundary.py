from dataProcessing.utils import read_nii_image, saveArray2nii
import numpy as np
import copy
import matplotlib.pyplot as plt
import os
import xlwt

## Organs name with L and R
organs_name_LR = {'1':'Brain Stem','3':'Mandible', '2':'Optical Chiasm',
                  '4':'Optical Nerve-L','5':'Optical Nerve-R','6':'Parotid glands-L',
                  '7':'Parotid glands-R','8':'Submandible glands-L','9':'Submandible glands-R'}

file_path = '/media/root/01456da2-1f1d-4b67-810e-b9cd3341133d/NPC_MICCAI_2015_original_data/HaN_2015_crop/train'
save_path = file_path + '_edge'
if os.path.exists(save_path) == 0:
    os.makedirs(save_path)
file_list = os.listdir(file_path)
R = 1
for fl in file_list:
    if 'label' not in fl:
        continue
    patient_id = fl.split('_')[0]
    print("Patient ID:", patient_id)
    mask_path = os.path.join(file_path, fl)

    mask = read_nii_image(mask_path)
    label_edge_final = np.zeros_like(mask)
    for i in range(len(organs_name_LR)):
        organs_id = i+1
        organ_name = organs_name_LR[str(organs_id)]
        print(organ_name)
        mask_one = (mask == organs_id)
        mask_one = mask_one.astype('float')

        if np.sum(mask_one) == 0:
            continue

        label_edge = copy.deepcopy(mask_one)
        edge_coord = np.argwhere(mask_one == 1)

        point_num = len(edge_coord)

        max_z = 0
        min_z = 999
        for en in range(point_num):
            edge_num_one = edge_coord[en]
            edge_num_one_min = copy.deepcopy(edge_num_one)
            # edge_num_one_max = copy.deepcopy(edge_num_one)
            for ll in range(len(edge_num_one_min)):
                edge_num_one_min[ll] = edge_num_one_min[ll] - 1
                edge_num_one_min[ll] = np.maximum(edge_num_one_min[ll], 0)

            min_gray = np.min(mask_one[edge_num_one[0],# - 1:edge_num_one[0] + 2,
                              edge_num_one_min[1]:edge_num_one[1] + 2,
                              edge_num_one_min[2]:edge_num_one[2] + 2])
            if edge_num_one[0] > max_z:
                max_z = edge_num_one[0]
            if edge_num_one[0] < min_z:
                min_z = edge_num_one[0]
            if min_gray != 0:
                label_edge[edge_num_one[0], edge_num_one[1], edge_num_one[2]] = 0

        label_edge[max_z, :, :] = mask_one[max_z, :, :]
        label_edge[min_z, :, :] = mask_one[min_z, :, :]
        label_edge_final = label_edge_final + label_edge * organs_id

    save_file_path = os.path.join(save_path, patient_id+'_0_edge.nii.gz')
    saveArray2nii(label_edge_final, save_file_path)