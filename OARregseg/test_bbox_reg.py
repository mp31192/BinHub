import os
import numpy as np
from dataProcessing.utils import read_nii_image
import xlwt
import medpy.metric.binary as medpyMetrics
from skimage.measure import label, regionprops

def Cal_dsc(predict,label):
    if np.count_nonzero(predict) > 0 and np.count_nonzero(label):
        intersection = predict*label
        tp = np.sum(intersection)
        pred_p = np.sum(predict)
        label_p = np.sum(label)
        dsc = 2*tp/(pred_p+label_p)
    elif np.sum(predict) == 0 and np.count_nonzero(label):
        dsc = 0
    else:
        dsc = -1
    return dsc

def Cal_sen(predict,label):
    if np.count_nonzero(predict) > 0 and np.count_nonzero(label):
        intersection = predict*label
        tp = np.sum(intersection)
        # pred_p = np.sum(predict)
        label_p = np.sum(label)
        sen = tp/(label_p)
    elif np.sum(predict) == 0 and np.count_nonzero(label):
        sen = 0
    else:
        sen = -1
    return sen

def Cal_pre(predict,label):
    if np.count_nonzero(predict) > 0 and np.count_nonzero(label):
        intersection = predict*label
        tp = np.sum(intersection)
        pred_p = np.sum(predict)
        # label_p = np.sum(label)
        pre = tp/(pred_p)
    elif np.sum(predict) == 0 and np.count_nonzero(label):
        pre = 0
    else:
        pre = -1
    return pre

def Cal_HD95(pred, target):
    pred = pred#.cpu().numpy()
    target = target#.cpu().numpy()
    if np.count_nonzero(pred) > 0 and np.count_nonzero(target):
        surDist1 = medpyMetrics.__surface_distances(pred, target, voxelspacing = [1.17, 1.17, 3])
        surDist2 = medpyMetrics.__surface_distances(target, pred, voxelspacing = [1.17, 1.17, 3])
        hd95 = np.percentile(np.hstack((surDist1, surDist2)), 95)
        return hd95
    elif np.sum(pred) == 0 and np.count_nonzero(target):
        return 99
    else:
        # Edge cases that medpy cannot handle
        return -1

def Cal_HD(pred, target):
    pred = pred  # .cpu().numpy()
    target = target  # .cpu().numpy()
    if np.count_nonzero(pred) > 0 and np.count_nonzero(target):
        surDist1 = medpyMetrics.__surface_distances(pred, target)
        surDist2 = medpyMetrics.__surface_distances(target, pred)
        hd = np.percentile(np.hstack((surDist1, surDist2)), 100)
        return hd
    elif np.sum(pred) == 0 and np.count_nonzero(target):
        return 99
    else:
        # Edge cases that medpy cannot handle
        return -1

# predict_ori_path = '/media/root/01456da2-1f1d-4b67-810e-b9cd3341133d/2019_MICCAI_challenge_NPC/predictbase_final'
# file_ori_list = os.listdir(predict_ori_path)
# for fol in file_ori_list:
#     if "2015" not in fol:
#         continue
#     predict_path = os.path.join(predict_ori_path, fol)
predict_path = '/media/root/01456da2-1f1d-4b67-810e-b9cd3341133d/2019_MICCAI_challenge_NPC/predictbase/202007291157_MICCAI2015OAR_registration_64channels_pad_2_221_reg_1_template2_resample_mask_result_noamp'
# mask_path = '/media/root/01456da2-1f1d-4b67-810e-b9cd3341133d/2019_MICCAI_challenge_NPC/HaN_OAR_crop/test'
mask_path = '/media/root/01456da2-1f1d-4b67-810e-b9cd3341133d/NPC_MICCAI_2015_original_data/HaN_2015_crop/test_all_headonly_resample_new'
excel_path = predict_path + '_result.xls'

file_list = os.listdir(mask_path)

## Organs name with L and R
organs_name_LR = {'1':'Brain Stem','3':'Mandible', '2':'Optical Chiasm',
                  '4':'Optical Nerve-L','5':'Optical Nerve-R','6':'Parotid glands-L',
                  '7':'Parotid glands-R','8':'Submandible glands-L','9':'Submandible glands-R'}
organs_LR_num = len(organs_name_LR)

workbook = xlwt.Workbook(encoding='ascii')
worksheetDSC = workbook.add_sheet('DSC')
worksheetDSC.write(0, 0, 'Name')
worksheetSEN = workbook.add_sheet('SEN_bbox_target')
worksheetSEN.write(0, 0, 'Name')
worksheetDSCbbox = workbook.add_sheet('DSC_bbox')
worksheetDSCbbox.write(0, 0, 'Name')


C = 0
for oi in range(1, organs_LR_num+1):
    on = organs_name_LR[str(oi)]
    C += 1
    worksheetDSC.write(0, C, on)
    worksheetSEN.write(0, C, on)
    worksheetDSCbbox.write(0, C, on)

R = 0

for fl in file_list:
    predict_name = fl
    if 'label.nii.gz' not in predict_name:
        continue
    print(predict_name)
    R += 1
    patient_id = predict_name.split('_')[0]
    worksheetDSC.write(R, 0, patient_id)
    worksheetSEN.write(R, 0, patient_id)
    worksheetDSCbbox.write(R, 0, patient_id)
    predict_result_path = os.path.join(predict_path, patient_id + '_0_affine_multichannel.nii.gz')
    mask_fullpath = os.path.join(mask_path, patient_id + '_0_label.nii.gz')
    labels = read_nii_image(mask_fullpath)
    labels = np.transpose(labels, [2, 1, 0])
    predict_result = read_nii_image(predict_result_path)
    predict_result = np.transpose(predict_result, [3, 2, 1, 0])

    count = 0
    for i in range(0, organs_LR_num):
        label_one = (labels == i + 1)
        label_one = label_one.astype('int')
        predict_one = predict_result[:, :, :, i]
        predict_one[predict_one >= 0.5] = 1
        predict_one[predict_one < 0.5] = 0
        predict_one = predict_one.astype('int')


        label_pros = regionprops(label_one, coordinates='rc')
        if len(label_pros) == 0:
            count += 1
            worksheetDSC.write(R, i + 1, str(-1))
            worksheetSEN.write(R, i + 1, str(-1))
            worksheetDSCbbox.write(R, i + 1, str(-1))
            continue
        label_bbox_coord = label_pros[0].bbox
        label_bbox_centroid = label_pros[0].centroid

        pred_pros = regionprops(predict_one, coordinates='rc')
        pred_bbox_coord = pred_pros[0].bbox
        pred_bbox_centroid = pred_pros[0].centroid

        label_bbox = np.zeros_like(label_one)
        pred_bbox = np.zeros_like(predict_one)

        # label_x1 = int(label_bbox_centroid[0] - (label_bbox_centroid[0] - label_bbox_coord[0]) * 1.2 - 4)
        # label_y1 = int(label_bbox_centroid[1] - (label_bbox_centroid[1] - label_bbox_coord[1]) * 1.2 - 4)
        # label_z1 = int(label_bbox_centroid[2] - (label_bbox_centroid[2] - label_bbox_coord[2]) * 1.2 - 4)
        # label_x2 = int(label_bbox_centroid[0] + (label_bbox_coord[3] - label_bbox_centroid[0]) * 1.2 + 4)
        # label_y2 = int(label_bbox_centroid[1] + (label_bbox_coord[4] - label_bbox_centroid[1]) * 1.2 + 4)
        # label_z2 = int(label_bbox_centroid[2] + (label_bbox_coord[5] - label_bbox_centroid[2]) * 1.2 + 4)
        #
        # label_bbox[label_x1:label_x2, label_y1:label_y2, label_z1:label_z2] = 1

        pred_shape = predict_one.shape
        label_x1 = np.maximum(int(pred_bbox_centroid[0] - (pred_bbox_centroid[0] - pred_bbox_coord[0]) * 1.2 - 4), 0)
        label_y1 = np.maximum(int(pred_bbox_centroid[1] - (pred_bbox_centroid[1] - pred_bbox_coord[1]) * 1.2 - 4), 0)
        label_z1 = np.maximum(int(pred_bbox_centroid[2] - (pred_bbox_centroid[2] - pred_bbox_coord[2]) * 1.2 - 4), 0)
        label_x2 = np.minimum(int(pred_bbox_centroid[0] + (pred_bbox_coord[3] - pred_bbox_centroid[0]) * 1.2 + 4), pred_shape[0])
        label_y2 = np.minimum(int(pred_bbox_centroid[1] + (pred_bbox_coord[4] - pred_bbox_centroid[1]) * 1.2 + 4), pred_shape[1])
        label_z2 = np.minimum(int(pred_bbox_centroid[2] + (pred_bbox_coord[5] - pred_bbox_centroid[2]) * 1.2 + 4), pred_shape[2])

        pred_bbox[label_x1:label_x2, label_y1:label_y2, label_z1:label_z2] = 1

        count += 1
        dsc_i = Cal_dsc(predict_one, label_one)
        sen_i = Cal_sen(predict_one, label_one)
        dsc_bbox_i = Cal_sen(pred_bbox, label_one)
        print(organs_name_LR[str(i+1)],' DSC:',dsc_i, " SEN:", sen_i, " DSC_bbox:", dsc_bbox_i)
        worksheetDSC.write(R, i + 1, str(dsc_i))
        worksheetSEN.write(R, i + 1, str(sen_i))
        worksheetDSCbbox.write(R, i + 1, str(dsc_bbox_i))

workbook.save(excel_path)
