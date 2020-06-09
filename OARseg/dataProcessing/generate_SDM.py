import os
from dataProcessing.utils import read_nii_image, saveArray2nii
import numpy as np
import copy
import time
import multiprocessing


def Cal_distance(image, mask):
    pass
    # surdist = medpyMetrics.__surface_distances(image, mask)
    # return surdist

def erode_3D(image, kernel):
    kernel_size = kernel.shape[0]
    d_image = np.zeros_like(image)
    center_move = int((kernel_size - 1)/2)
    for z in range(center_move, image.shape[0] - kernel_size + 1):
        for i in range(center_move, image.shape[1] - kernel_size + 1):
            for j in range(center_move, image.shape[2] - kernel_size + 1):
                d_image[z, i, j] = np.min(image[z - center_move:z + center_move,
                                          i - center_move:i + center_move,
                                          j - center_move:j + center_move])
    return d_image

def dilate_3D(image, kernel):
    kernel_size = kernel.shape[0]
    d_image = np.zeros_like(image)
    center_move = int((kernel_size - 1)/2)
    for z in range(center_move, image.shape[0] - kernel_size + 1):
        for i in range(center_move, image.shape[1] - kernel_size + 1):
            for j in range(center_move, image.shape[2] - kernel_size + 1):
                d_image[z, i, j] = np.max(image[z - center_move:z + center_move,
                                          i - center_move:i + center_move,
                                          j - center_move:j + center_move])
    return d_image

def DanielssonCal(image):
    shape_z, shape_y, shape_x = np.shape(image)
    L = np.zeros([shape_z, shape_y, shape_x, 3])
    L[:, :, :, 0] = L[:, :, :, 0] + image
    L[:, :, :, 1] = L[:, :, :, 1] + image
    L[:, :, :, 2] = L[:, :, :, 2] + image
    L[L == 0] = 1000
    L[L == 1] = 0

    for sz in range(0,shape_z - 1):
        start_time = time.time()
        if np.min(L[sz,:,:,:]) != 0:
            continue
        for j in range(1,shape_y):
            for i in range(0,shape_x - 1):
                point1 = [L[sz, j, i, 1], L[sz, j - 1, i, 1] + 1]
                point2 = [L[sz, j, i, 2], L[sz, j - 1, i, 2]]
                min_p = np.argmin(np.array([L[sz, j, i, 1] + L[sz, j, i, 2], L[sz, j - 1, i, 1] + 1 + L[sz, j - 1, i, 2]]))
                L[sz, j, i, 1] = point1[min_p]
                L[sz, j, i, 2] = point2[min_p]
            for i in range(1,shape_x - 1):
                point1 = [L[sz, j, i, 1], L[sz, j, i - 1, 1]]
                point2 = [L[sz, j, i, 2], L[sz, j, i - 1, 2] + 1]
                min_p = np.argmin(np.array([L[sz, j, i, 1] + L[sz, j, i, 2], L[sz, j, i - 1, 2] + 1 + L[sz, j, i - 1, 1]]))
                L[sz, j, i, 1] = point1[min_p]
                L[sz, j, i, 2] = point2[min_p]
            for i in range(shape_x - 2, -1, -1):
                point1 = [L[sz, j, i, 1], L[sz, j, i + 1, 1]]
                point2 = [L[sz, j, i, 2], L[sz, j, i + 1, 2] + 1]
                min_p = np.argmin(np.array([L[sz, j, i, 1] + L[sz, j, i, 2], L[sz, j, i + 1, 2] + 1 + L[sz, j, i + 1, 1]]))
                L[sz, j, i, 1] = point1[min_p]
                L[sz, j, i, 2] = point2[min_p]
                # L[sz, j, i, 1] = np.minimum(L[sz, j, i, 1], L[sz, j, i + 1, 1])
        for j in range(shape_y - 2, -1, -1):
            for i in range(1, shape_x - 1):
                point1 = [L[sz, j, i, 1], L[sz, j + 1, i, 1] + 1]
                point2 = [L[sz, j, i, 2], L[sz, j + 1, i, 2]]
                min_p = np.argmin(np.array([L[sz, j, i, 1] + L[sz, j, i, 2], L[sz, j + 1, i, 1] + 1 + L[sz, j + 1, i, 2]]))
                L[sz, j, i, 1] = point1[min_p]
                L[sz, j, i, 2] = point2[min_p]
            for i in range(1, shape_x):
                point1 = [L[sz, j, i, 1], L[sz, j, i - 1, 1]]
                point2 = [L[sz, j, i, 2], L[sz, j, i - 1, 2] + 1]
                min_p = np.argmin(np.array([L[sz, j, i, 1] + L[sz, j, i, 2], L[sz, j, i - 1, 2] + 1 + L[sz, j, i - 1, 1]]))
                L[sz, j, i, 1] = point1[min_p]
                L[sz, j, i, 2] = point2[min_p]
            for i in range(shape_x - 2, -1, -1):
                point1 = [L[sz, j, i, 1], L[sz, j, i + 1, 1]]
                point2 = [L[sz, j, i, 2], L[sz, j, i + 1, 2] + 1]
                min_p = np.argmin(np.array([L[sz, j, i, 1] + L[sz, j, i, 2], L[sz, j, i + 1, 2] + 1 + L[sz, j, i + 1, 1]]))
                L[sz, j, i, 1] = point1[min_p]
                L[sz, j, i, 2] = point2[min_p]
        print("spend time:",time.time() - start_time)
    return L

FilePath = '/media/root/01456da2-1f1d-4b67-810e-b9cd3341133d/2019_MICCAI_challenge_NPC/HaN_OAR_crop/train'
file_list = os.listdir(FilePath)

SavePath = '/media/root/01456da2-1f1d-4b67-810e-b9cd3341133d/2019_MICCAI_challenge_NPC/HaN_OAR_crop/train_dist'

for fl in file_list:
    if 'label.nii' in fl:
        label_path = os.path.join(FilePath, fl)
        patient_id = fl.split('_')[0]
        print("Patient ID: ", patient_id)

        label = read_nii_image(label_path)
        shape_z, shape_y, shape_x = np.shape(label)
        label_all = np.zeros([shape_z, shape_y, shape_x, 22])
        for i in range(1, 23):
            print("Organs ID:", i)
            if i == 6 or i == 7:
                label_i = (label == i) + (label == i + 1)
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
        dist_path = os.path.join(SavePath, patient_id + '_' + str(i) +'.nii')
        saveArray2nii(label_all, dist_path)

        print("MUDA!!")