import os
import shutil

reg_result_path = "/media/root/01456da2-1f1d-4b67-810e-b9cd3341133d/2019_MICCAI_challenge_NPC/predictbase/202007061453_MICCAI2015OAR_registration_OneCycle_alltemplate2_scores_map_my_deform_ratioresize_multWL_train"
seg_path = "/media/root/01456da2-1f1d-4b67-810e-b9cd3341133d/NPC_MICCAI_2015_original_data/HaN_2015_crop/train_all_headonly_noresample_new"

reg_result_list = os.listdir(reg_result_path)
seg_list = os.listdir(seg_path)

for sl in seg_list:
    if "label.nii.gz" in sl:
        patient_id = sl.split('_')[0]
        print(patient_id)
        reg_result_filename = patient_id + "_0_result_affine.nii.gz"
        new_reg_result_filename = patient_id + "_0_affine.nii.gz"

        old_path = os.path.join(reg_result_path, reg_result_filename)
        new_path = os.path.join(seg_path, new_reg_result_filename)

        shutil.copy(old_path, new_path)