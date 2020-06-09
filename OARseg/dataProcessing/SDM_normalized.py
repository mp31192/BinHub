import os
from dataProcessing.utils import read_nii_image, saveArray2nii
import numpy as np
import copy
import time

OriPath = '/media/root/01456da2-1f1d-4b67-810e-b9cd3341133d/2019_MICCAI_challenge_NPC/HaN_OAR_crop/val'

FilePath = '/media/root/01456da2-1f1d-4b67-810e-b9cd3341133d/2019_MICCAI_challenge_NPC/HaN_OAR_crop/val_dist_3D'
file_list = os.listdir(FilePath)

SavePath = '/media/root/01456da2-1f1d-4b67-810e-b9cd3341133d/2019_MICCAI_challenge_NPC/HaN_OAR_crop/val_dist_tanh_normalized'
if os.path.exists(SavePath) == 0:
    os.makedirs(SavePath)

for fl in file_list:
    dist_path = os.path.join(FilePath, fl)
    patient_id = fl.split('_')[0]
    print("Patient ID: ", patient_id)

    dist_map = read_nii_image(dist_path)
    shape_z, shape_y, shape_x, channels = np.shape(dist_map)
    # dist_normalied_map = np.zeros_like(dist_map)
    label_name = patient_id + '_0_label.nii'
    label_path = os.path.join(OriPath, label_name)
    label = read_nii_image(label_path)
    start_time = time.time()
    for i in range(1, 23):

        print("Organs ID:", i)
        if i == 6 or i == 7:
            label_one = (label == 6) + (label == 7)
        else:
            label_one = (label == i)
        label_one = label_one.astype('int')

        label_one_bg = copy.deepcopy(label_one)
        label_one_bg = 1 - label_one_bg
        label_one_fg = label_one

        dist_map_one = copy.deepcopy(dist_map[:,:,:,i-1])

        ## min max
        # dist_map_fg = copy.deepcopy(dist_map_one)
        # dist_map_bg = copy.deepcopy(dist_map_one)
        #
        # dist_map_fg = dist_map_fg * label_one_fg
        # dist_map_bg = dist_map_bg * label_one_bg
        #
        # min_fg = np.abs(np.min(dist_map_fg))
        # max_bg = np.abs(np.max(dist_map_bg))
        # if min_fg == 0:
        #     min_fg = 1

        # dist_map_fg = dist_map_fg / min_fg
        # dist_map_bg = dist_map_bg / max_bg

        # dist_normalied_map = dist_map_fg + dist_map_bg

        ## tanh
        dist_normalied_map = np.tanh(dist_map_one)
        print("Max map:",np.max(dist_normalied_map),"Min map:",np.min(dist_normalied_map))

        dist_path = os.path.join(SavePath, patient_id + '_0_' + str(i) + '_dist.nii.gz')
        saveArray2nii(dist_normalied_map, dist_path)
    print("Patient Time:",time.time() - start_time)
    print("MUDA!!")