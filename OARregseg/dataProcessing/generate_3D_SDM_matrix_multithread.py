import os
from dataProcessing.utils import read_nii_image, saveArray2nii
import numpy as np
import copy
import time
import multiprocessing

def process(label_path):
    filename = label_path.split('/')[-1]
    patient_id = filename.split('_')[0]
    print("Patient ID: ", label_path)
    label = read_nii_image(label_path)
    shape_z, shape_y, shape_x = np.shape(label)
    label_all = np.zeros([shape_z, shape_y, shape_x, 9])
    for i in range(1, 10):
        print("Organs ID:", i)
        if i == 4 or i == 5:
            label_i = (label == 4) + (label == 5)
        else:
            label_i = (label == i)

        label_i = label_i.astype('int8')

        # dist_path = os.path.join(SavePath, patient_id + '_' + str(i) + '_ori.nii')
        # saveArray2nii(label_i, dist_path)

        label_edge = copy.deepcopy(label_i)
        edge_coord = np.argwhere(label_edge == 1)

        edge_num = len(edge_coord)

        if edge_num == 0:
            continue

        max_z = 0
        min_z = 999
        for en in range(edge_num):
            edge_num_one = edge_coord[en]
            min_gray = np.min(label_i[edge_num_one[0],# - 1:edge_num_one[0] + 2,
                              edge_num_one[1] - 1:edge_num_one[1] + 2,
                              edge_num_one[2] - 1:edge_num_one[2] + 2])
            if edge_num_one[0] > max_z:
                max_z = edge_num_one[0]
            if edge_num_one[0] < min_z:
                min_z = edge_num_one[0]
            if min_gray != 0:
                label_edge[edge_num_one[0], edge_num_one[1], edge_num_one[2]] = 0

        label_edge[max_z, :, :] = label_i[max_z, :, :]
        label_edge[min_z, :, :] = label_i[min_z, :, :]

        label_inside = copy.deepcopy(label_i)
        label_inside[label_inside == 1] = -1
        label_inside[label_inside == 0] = 1

        dist_mask = DanielssonCal(label_edge)
        dist_mask[dist_mask == 1000] = 0
        dist_mask_f = np.sqrt(dist_mask[:, :, :, 0] ** 2 + dist_mask[:, :, :, 1] ** 2 + dist_mask[:, :, :, 2] ** 2)

        dist_mask_f = dist_mask_f * label_inside
        label_all[:, :, :, i - 1] = dist_mask_f
    dist_path = os.path.join(SavePath, patient_id + '_0_dist.nii.gz')
    saveArray2nii(label_all, dist_path)

def DanielssonCal(image):
    shape_z, shape_y, shape_x = np.shape(image)
    L = np.zeros([shape_z, shape_y, shape_x, 3])
    L[:, :, :, 0] = L[:, :, :, 0] + image
    L[:, :, :, 1] = L[:, :, :, 1] + image
    L[:, :, :, 2] = L[:, :, :, 2] + image
    L[L == 0] = 1000
    L[L == 1] = 0

    start_time = time.time()
    for sz in range(1,shape_z):
        point0 = [L[sz, :, :, 0], L[sz - 1, :, :, 0] + 1]
        point1 = [L[sz, :, :, 1], L[sz - 1, :, :, 1]]
        point2 = [L[sz, :, :, 2], L[sz - 1, :, :, 2]]
        min_p = np.argmin(np.array(
            [np.sqrt(np.square(L[sz, :, :, 0]) + np.square(L[sz, :, :, 1]) + np.square(L[sz, :, :, 2])),
             np.square(L[sz - 1, :, :, 0] + 1) + np.sqrt(
                 np.square(L[sz - 1, :, :, 1]) + np.square(L[sz - 1, :, :, 2]))]), axis=0)
        for j in range(0, shape_y):
            for i in range(0, shape_x):
                point0_i = point0[min_p[j][i]]
                point1_i = point1[min_p[j][i]]
                point2_i = point2[min_p[j][i]]
                L[sz, j, i, 0] = point0_i[j][i]
                L[sz, j, i, 1] = point1_i[j][i]
                L[sz, j, i, 2] = point2_i[j][i]

    for j in range(1, shape_y):
        point0 = [L[:, j, :, 0], L[:, j - 1, :, 0]]
        point1 = [L[:, j, :, 1], L[:, j - 1, :, 1] + 1]
        point2 = [L[:, j, :, 2], L[:, j - 1, :, 2]]
        min_p = np.argmin(np.array(
            [np.sqrt(np.square(L[:, j, :, 0]) + np.square(L[sz, j, :, 1]) + np.square(L[:, j, :, 2])),
             np.sqrt(np.square(L[:, j - 1, :, 0]) + np.square(L[:, j - 1, :, 1] + 1) + np.square(L[:, j - 1, :, 2]))]), axis=0)
        for sz in range(0, shape_z):
            for i in range(0, shape_x):
                point0_i = point0[min_p[sz][i]]
                point1_i = point1[min_p[sz][i]]
                point2_i = point2[min_p[sz][i]]
                L[sz, j, i, 0] = point0_i[sz][i]
                L[sz, j, i, 1] = point1_i[sz][i]
                L[sz, j, i, 2] = point2_i[sz][i]

    for i in range(1, shape_x):
        point0 = [L[:, :, i, 0], L[:, :, i - 1, 0]]
        point1 = [L[:, :, i, 1], L[:, :, i - 1, 1]]
        point2 = [L[:, :, i, 2], L[:, :, i - 1, 2] + 1]
        min_p = np.argmin(np.array(
            [np.sqrt(np.square(L[:, :, i, 0]) + np.square(L[:, :, i, 1]) + np.square(L[:, :, i, 2])),
             np.sqrt(np.square(L[:, :, i - 1, 0]) + np.square(L[:, :, i - 1, 2] + 1) + np.square(
                 L[:, :, i - 1, 1]))]), axis=0)
        for sz in range(0,shape_z):
            for j in range(0, shape_y):
                point0_i = point0[min_p[sz][j]]
                point1_i = point1[min_p[sz][j]]
                point2_i = point2[min_p[sz][j]]
                L[sz, j, i, 0] = point0_i[sz][j]
                L[sz, j, i, 1] = point1_i[sz][j]
                L[sz, j, i, 2] = point2_i[sz][j]

    for i in range(shape_x - 2, -1, -1):
        point0 = [L[:, :, i, 0], L[:, :, i + 1, 0]]
        point1 = [L[:, :, i, 1], L[:, :, i + 1, 1]]
        point2 = [L[:, :, i, 2], L[:, :, i + 1, 2] + 1]
        min_p = np.argmin(np.array(
            [np.sqrt(np.square(L[:, :, i, 0]) + np.square(L[:, :, i, 1]) + np.square(L[:, :, i, 2])),
             np.sqrt(np.square(L[:, :, i + 1, 0]) + np.square(L[:, :, i + 1, 2] + 1) + np.square(
                 L[:, :, i + 1, 1]))]), axis=0)
        for sz in range(0,shape_z):
            for j in range(0, shape_y):
                point0_i = point0[min_p[sz][j]]
                point1_i = point1[min_p[sz][j]]
                point2_i = point2[min_p[sz][j]]
                L[sz, j, i, 0] = point0_i[sz][j]
                L[sz, j, i, 1] = point1_i[sz][j]
                L[sz, j, i, 2] = point2_i[sz][j]

    for sz in range(shape_z - 2, -1, -1):
        point0 = [L[sz, :, :, 0], L[sz + 1, :, :, 0] + 1]
        point1 = [L[sz, :, :, 1], L[sz + 1, :, :, 1]]
        point2 = [L[sz, :, :, 2], L[sz + 1, :, :, 2]]
        min_p = np.argmin(np.array(
            [np.sqrt(np.square(L[sz, :, :, 0]) + np.square(L[sz, :, :, 1]) + np.square(L[sz, :, :, 2])),
             np.sqrt(np.square(L[sz + 1, :, :, 0] + 1) + np.square(L[sz + 1, :, :, 1]) + np.square(L[sz + 1, :, :, 2]))]), axis=0)
        for j in range(0, shape_y):
            for i in range(0, shape_x):
                point0_i = point0[min_p[j][i]]
                point1_i = point1[min_p[j][i]]
                point2_i = point2[min_p[j][i]]
                L[sz, j, i, 0] = point0_i[j][i]
                L[sz, j, i, 1] = point1_i[j][i]
                L[sz, j, i, 2] = point2_i[j][i]

    for j in range(shape_y - 2, -1, -1):
        point0 = [L[:, j, :, 0], L[:, j + 1, :, 0]]
        point1 = [L[:, j, :, 1], L[:, j + 1, :, 1] + 1]
        point2 = [L[:, j, :, 2], L[:, j + 1, :, 2]]
        min_p = np.argmin(np.array(
            [np.sqrt(np.square(L[:, j, :, 0]) + np.square(L[:, j, :, 1]) + np.square(L[:, j, :, 2])),
             np.sqrt(np.square(L[:, j + 1, :, 0]) + np.square(L[:, j + 1, :, 1] + 1) + np.square(L[:, j + 1, :, 2]))]), axis=0)
        for sz in range(0, shape_z):
            for i in range(0, shape_x):
                point0_i = point0[min_p[sz][i]]
                point1_i = point1[min_p[sz][i]]
                point2_i = point2[min_p[sz][i]]
                L[sz, j, i, 0] = point0_i[sz][i]
                L[sz, j, i, 1] = point1_i[sz][i]
                L[sz, j, i, 2] = point2_i[sz][i]

    for i in range(1, shape_x):
        point0 = [L[:, :, i, 0], L[:, :, i - 1, 0]]
        point1 = [L[:, :, i, 1], L[:, :, i - 1, 1]]
        point2 = [L[:, :, i, 2], L[:, :, i - 1, 2] + 1]
        min_p = np.argmin(np.array(
            [np.sqrt(np.square(L[:, :, i, 0]) + np.square(L[:, :, i, 1]) + np.square(L[:, :, i, 2])),
             np.sqrt(np.square(L[:, :, i - 1, 0]) + np.square(L[:, :, i - 1, 2] + 1) + np.square(
                 L[:, :, i - 1, 1]))]), axis=0)
        for sz in range(0, shape_z):
            for j in range(0, shape_y):
                point0_i = point0[min_p[sz][j]]
                point1_i = point1[min_p[sz][j]]
                point2_i = point2[min_p[sz][j]]
                L[sz, j, i, 0] = point0_i[sz][j]
                L[sz, j, i, 1] = point1_i[sz][j]
                L[sz, j, i, 2] = point2_i[sz][j]

    for i in range(shape_x - 2, -1, -1):
        point0 = [L[:, :, i, 0], L[:, :, i + 1, 0]]
        point1 = [L[:, :, i, 1], L[:, :, i + 1, 1]]
        point2 = [L[:, :, i, 2], L[:, :, i + 1, 2] + 1]
        min_p = np.argmin(np.array(
            [np.sqrt(np.square(L[:, :, i, 0]) + np.square(L[:, :, i, 1]) + np.square(L[:, :, i, 2])),
             np.sqrt(np.square(L[:, :, i + 1, 0]) + np.square(L[:, :, i + 1, 2] + 1) + np.square(
                 L[:, :, i + 1, 1]))]), axis=0)
        for sz in range(0, shape_z):
            for j in range(0, shape_y):
                point0_i = point0[min_p[sz][j]]
                point1_i = point1[min_p[sz][j]]
                point2_i = point2[min_p[sz][j]]
                L[sz, j, i, 0] = point0_i[sz][j]
                L[sz, j, i, 1] = point1_i[sz][j]
                L[sz, j, i, 2] = point2_i[sz][j]
    #
    print("Spend time:", time.time() - start_time)
    return L

FilePath = '/media/root/01456da2-1f1d-4b67-810e-b9cd3341133d/NPC_MICCAI_2015_original_data/HaN_2015_crop/test_all_headonly_noresample_new'
file_list = os.listdir(FilePath)

SavePath = '/media/root/01456da2-1f1d-4b67-810e-b9cd3341133d/NPC_MICCAI_2015_original_data/HaN_2015_crop/test_all_headonly_noresample_new_SDM'
if os.path.exists(SavePath) == 0:
    os.makedirs(SavePath)

label_path_list = []
for fl in file_list:
    if 'label.nii' in fl:
        label_path = os.path.join(FilePath, fl)
        patient_id = fl.split('_')[0]
        label_path_list.append(label_path)

p1 = multiprocessing.Pool(10)
b = p1.map(process, label_path_list)
p1.close()
p1.join()

print("MUDA!!")