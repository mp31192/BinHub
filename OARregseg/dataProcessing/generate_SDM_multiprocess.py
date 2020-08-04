import os
from dataProcessing.utils import read_nii_image, saveArray2nii
import numpy as np
import copy
import time
import multiprocessing

def Process(L):
    shape_z, shape_y, shape_x, channels = np.shape(L)
    sz = 0
    for j in range(1, shape_y):
        for i in range(1, shape_x - 1):
            point1 = [L[sz, j, i, 1], L[sz, j - 1, i, 1] + 1]
            point2 = [L[sz, j, i, 2], L[sz, j - 1, i, 2]]
            min_p = np.argmin(np.array([np.sqrt(L[sz, j, i, 1] ** 2 + L[sz, j, i, 2] ** 2),
                                        np.sqrt(np.square(L[sz, j - 1, i, 1] + 1) + np.square(L[sz, j - 1, i, 2]))]))
            L[sz, j, i, 1] = point1[min_p]
            L[sz, j, i, 2] = point2[min_p]
        for i in range(1, shape_x - 1):
            point1 = [L[sz, j, i, 1], L[sz, j, i - 1, 1]]
            point2 = [L[sz, j, i, 2], L[sz, j, i - 1, 2] + 1]
            min_p = np.argmin(np.array([np.sqrt(np.square(L[sz, j, i, 1]) + np.square(L[sz, j, i, 2])),
                                        np.sqrt(np.square(L[sz, j, i - 1, 2] + 1) + np.square(L[sz, j, i - 1, 1]))]))
            L[sz, j, i, 1] = point1[min_p]
            L[sz, j, i, 2] = point2[min_p]
        for i in range(shape_x - 2, -1, -1):
            point1 = [L[sz, j, i, 1], L[sz, j, i + 1, 1]]
            point2 = [L[sz, j, i, 2], L[sz, j, i + 1, 2] + 1]
            min_p = np.argmin(np.array([np.sqrt(np.square(L[sz, j, i, 1]) + np.square(L[sz, j, i, 2])),
                                        np.sqrt(np.square(L[sz, j, i + 1, 2] + 1) + np.square(L[sz, j, i + 1, 1]))]))
            L[sz, j, i, 1] = point1[min_p]
            L[sz, j, i, 2] = point2[min_p]
    return L

def Process2(L):
    shape_z, shape_y, shape_x, channels = np.shape(L)
    sz = 0
    for j in range(shape_y - 2, -1, -1):
        for i in range(1, shape_x - 1):
            point1 = [L[sz, j, i, 1], L[sz, j + 1, i, 1] + 1]
            point2 = [L[sz, j, i, 2], L[sz, j + 1, i, 2]]
            min_p = np.argmin(np.array([np.sqrt(np.square(L[sz, j, i, 1]) + np.square(L[sz, j, i, 2])),
                                        np.sqrt(np.square(L[sz, j + 1, i, 1] + 1) + np.square(L[sz, j + 1, i, 2]))]))
            L[sz, j, i, 1] = point1[min_p]
            L[sz, j, i, 2] = point2[min_p]
        for i in range(1, shape_x):
            point1 = [L[sz, j, i, 1], L[sz, j, i - 1, 1]]
            point2 = [L[sz, j, i, 2], L[sz, j, i - 1, 2] + 1]
            min_p = np.argmin(np.array([np.sqrt(np.square(L[sz, j, i, 1]) + np.square(L[sz, j, i, 2])),
                                        np.sqrt(np.square(L[sz, j, i - 1, 2] + 1) + np.square(L[sz, j, i - 1, 1]))]))
            L[sz, j, i, 1] = point1[min_p]
            L[sz, j, i, 2] = point2[min_p]
        for i in range(shape_x - 2, -1, -1):
            point1 = [L[sz, j, i, 1], L[sz, j, i + 1, 1]]
            point2 = [L[sz, j, i, 2], L[sz, j, i + 1, 2] + 1]
            min_p = np.argmin(np.array([np.sqrt(np.square(L[sz, j, i, 1]) + np.square(L[sz, j, i, 2])),
                                        np.sqrt(np.square(L[sz, j, i + 1, 2] + 1) + np.square(L[sz, j, i + 1, 1]))]))
            L[sz, j, i, 1] = point1[min_p]
            L[sz, j, i, 2] = point2[min_p]
    return L

def DanielssonCal(image):
    shape_z, shape_y, shape_x = np.shape(image)
    L = np.zeros([shape_z, shape_y, shape_x, 3])
    L[:, :, :, 0] = L[:, :, :, 0] + image
    L[:, :, :, 1] = L[:, :, :, 1] + image
    L[:, :, :, 2] = L[:, :, :, 2] + image
    L[L == 0] = 1000
    L[L == 1] = 0

    param = []
    z_list = []
    for sz in range(0,shape_z - 1):
        start_time = time.time()
        if np.min(L[sz,:,:,:]) != 0:
            continue
        param.append(L[sz:sz+1, :, :, :])
        z_list.append(sz)

    L_f = np.zeros([shape_z, shape_y, shape_x, 3])

    p1 = multiprocessing.Pool(32)
    b = p1.map(Process, param)
    p1.close()
    p1.join()
    p2 = multiprocessing.Pool(32)
    b2 = p2.map(Process2, b)
    p2.close()
    p2.join()
    print("spend time:", time.time() - start_time)
    count = 0
    for zl in z_list:
        L_f[zl:zl+1, :, :, :] = b2[count]
        count += 1
    return L_f

FilePath = '/media/root/01456da2-1f1d-4b67-810e-b9cd3341133d/2019_MICCAI_challenge_NPC/HaN_OAR_crop/val'
file_list = os.listdir(FilePath)

SavePath = '/media/root/01456da2-1f1d-4b67-810e-b9cd3341133d/2019_MICCAI_challenge_NPC/HaN_OAR_crop/val_dist'
if os.path.exists(SavePath) == 0:
    os.makedirs(SavePath)

for fl in file_list:
    if 'label.nii' in fl:
        label_path = os.path.join(FilePath, fl)
        patient_id = fl.split('_')[0]
        print("Patient ID: ", patient_id)

        label = read_nii_image(label_path)
        shape_z, shape_y, shape_x = np.shape(label)
        label_all = np.zeros([shape_z, shape_y, shape_x, 22])
        start_time = time.time()
        for i in range(1, 23):
            print("Organs ID:", i)
            if i == 6 or i == 7:
                label_i = (label == 6) + (label == 7)
            else:
                label_i = (label == i)

            label_i = label_i.astype('int8')

            # dist_path = os.path.join(SavePath, patient_id + '_' + str(i) + '_ori.nii')
            # saveArray2nii(label_i, dist_path)

            label_edge = copy.deepcopy(label_i)
            edge_coord = np.argwhere(label_edge==1)

            edge_num = len(edge_coord)

            for en in range(edge_num):
                edge_num_one = edge_coord[en]
                min_gray = np.min(label_i[edge_num_one[0],# - 1:edge_num_one[0] + 2,
                                  edge_num_one[1] - 1:edge_num_one[1] + 2,
                                  edge_num_one[2] - 1:edge_num_one[2] + 2])
                if min_gray != 0:
                    label_edge[edge_num_one[0],edge_num_one[1],edge_num_one[2]] = 0

            label_inside = copy.deepcopy(label_i)
            label_inside[label_inside == 1] = -1
            label_inside[label_inside == 0] = 1

            dist_mask = DanielssonCal(label_edge)
            dist_mask[dist_mask == 1000] = 0
            dist_mask_f = np.sqrt(dist_mask[:,:,:,1] ** 2 + dist_mask[:,:,:,2] ** 2)

            dist_mask_f = dist_mask_f * label_inside
            label_all[:,:,:,i-1] = dist_mask_f
        dist_path = os.path.join(SavePath, patient_id + '_dist.nii.gz')
        saveArray2nii(label_all, dist_path)
        print("Patient Time:",time.time() - start_time)
        print("MUDA!!")