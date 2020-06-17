import os
import numpy as np
from dataProcessing.utils import read_nii_image
import xlwt
import medpy.metric.binary as medpyMetrics

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
predict_path = '/media/root/01456da2-1f1d-4b67-810e-b9cd3341133d/2019_MICCAI_challenge_NPC/predictbase/202006151013_MICCAI2015OAR_segmentation_CycleOne_noresample_context_1p2crop_2down_randint'
# mask_path = '/media/root/01456da2-1f1d-4b67-810e-b9cd3341133d/2019_MICCAI_challenge_NPC/HaN_OAR_crop/test'
mask_path = '/media/root/01456da2-1f1d-4b67-810e-b9cd3341133d/NPC_MICCAI_2015_original_data/HaN_2015_crop/test_all_headonly_noresample_new'
excel_path = predict_path + '_result.xls'

file_list = os.listdir(predict_path)

## Organs name with L and R
organs_name_LR = {'1':'Brain Stem','3':'Mandible', '2':'Optical Chiasm',
                  '4':'Optical Nerve-L','5':'Optical Nerve-R','6':'Parotid glands-L',
                  '7':'Parotid glands-R','8':'Submandible glands-L','9':'Submandible glands-R'}
organs_LR_num = len(organs_name_LR)

workbook = xlwt.Workbook(encoding='ascii')
worksheetDSC = workbook.add_sheet('DSC')
worksheetDSC.write(0, 0, 'Name')
worksheetSEN = workbook.add_sheet('SEN')
worksheetSEN.write(0, 0, 'Name')
worksheetPRE = workbook.add_sheet('PRE')
worksheetPRE.write(0, 0, 'Name')
worksheet95HD = workbook.add_sheet('95HD')
worksheet95HD.write(0, 0, 'Name')
worksheetHD = workbook.add_sheet('HD')
worksheetHD.write(0, 0, 'Name')


C = 0
for oi in range(1, organs_LR_num+1):
    on = organs_name_LR[str(oi)]
    C += 1
    worksheetDSC.write(0, C, on)
    worksheet95HD.write(0, C, on)

R = 0

for fl in file_list:
    predict_name = fl
    if 'result.nii.gz' not in predict_name:
        continue
    print(predict_name)
    R += 1
    patient_id = predict_name.split('_')[0]
    worksheetDSC.write(R, 0, patient_id)
    worksheetSEN.write(R, 0, patient_id)
    worksheetPRE.write(R, 0, patient_id)
    worksheet95HD.write(R, 0, patient_id)
    worksheetHD.write(R, 0, patient_id)
    predict_result_path = os.path.join(predict_path, patient_id + '_result.nii.gz')
    mask_fullpath = os.path.join(mask_path, patient_id + '_0_label.nii.gz')
    labels = read_nii_image(mask_fullpath)
    labels = np.transpose(labels, [2, 1, 0])
    predict_result = read_nii_image(predict_result_path)
    predict_result = np.transpose(predict_result, [2, 1, 0])

    count = 0
    for i in range(0, organs_LR_num):
        label_one = (labels == i + 1)
        label_one = label_one.astype('int')
        predict_one = (predict_result == count + 1)
        predict_one = predict_one.astype('int')
        count += 1
        dsc_i = Cal_dsc(predict_one, label_one)
        # sen_i = Cal_sen(predict_one, label_one)
        # pre_i = Cal_pre(predict_one, label_one)
        # hd_i = Cal_HD(predict_one, label_one)
        hd95_i = Cal_HD95(predict_one, label_one)
        print(organs_name_LR[str(i+1)],' DSC:',dsc_i, " 95HD:", hd95_i)#, ' SEN:', sen_i, ' PRE:', pre_i)
        worksheetDSC.write(R, i + 1, str(dsc_i))
        # worksheetSEN.write(R, i + 1, str(sen_i))
        # worksheetPRE.write(R, i + 1, str(pre_i))
        # worksheetHD.write(R, i + 1, str(hd_i))
        worksheet95HD.write(R, i + 1, str(hd95_i))
        # dsc_arr.append(dsc_i)

workbook.save(excel_path)
# workbook95HD.save(excel_95hd_path)