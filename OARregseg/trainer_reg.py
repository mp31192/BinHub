import torch
import RegDatasetLoader as DatasetLoader
import registrater_seg_new as registrater
import systemsetup
import shutil
import os
import torch.optim as optim
import time
import experiments.Registration_Bin_OARs as expConfig
import xlwt,xlrd
from RAdam import RAdam

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

organs_combine = {'Brain Stem':[1],
                  'Optical Nerve Chiasm':[2,4],
                  'Parotid glands-L':[5],'Parotid glands-R':[6],
                  'Mandible':[3], 'Submandible glands-L':[7], 'Submandible glands-R':[8]}

organs_model = {'Brain Stem':0,
                'Optical Nerve Chiasm':1,
                'Parotid glands-L':2,
                'Mandible':3,
                'Submandible glands-L':4}

organs_num = len(organs_model)
organs_model_name = list(organs_model.keys())

def main():

    ## task name, what dataset
    task_name = 'MICCAI2015OAR'

    ## labels id numbers
    mask_num = 9

    ## experiment time
    exp_time = time.strftime("%Y%m%d%H%M", time.localtime(time.time()))

    ## experiment operation
    exp_opt = "registration_64channels_pad_2_221_reg_1_template2_resample_RAdam_amp"

    ## experiment name
    expConfig.id = exp_time + '_' + task_name + '_' + exp_opt if exp_opt else exp_time + '_' + task_name
    print("EXP ID:", expConfig.id)
    ## excel for logging
    expConfig.EXCEL_ID = 'training_info.xls'

    expConfig.PREDICT = False
    expConfig.RESTORE_ID = "202007291157_MICCAI2015OAR_registration_64channels_pad_2_221_reg_1_template2_resample_mask_result_noamp"
    expConfig.MODEL_NAME = "best_model"

    AUTO_FIND_LR = False

    expConfig.INITIAL_LR = 5e-5
    MAX_LR = 5e-4
    expConfig.L2_REGULARIZER = 1e-5

    TRAIN_PATH = os.path.join(systemsetup.DATA_PATH, 'train_all_headonly_resample_new')
    VAL_PATH = os.path.join(systemsetup.DATA_PATH, 'test_all_headonly_resample_new')
    TEST_PATH = os.path.join(systemsetup.DATA_PATH, 'test_all_headonly_resample_new')

    # load data
    trainset = DatasetLoader.OARDataset(TRAIN_PATH, expConfig, template_id='6', mask_num=mask_num, mode="train", hasMasks=True,
                                        AutoChangeDim=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=expConfig.BATCH_SIZE, shuffle=True, pin_memory=False,
                                              num_workers=expConfig.DATASET_WORKERS)

    valset = DatasetLoader.OARDataset(VAL_PATH, expConfig, template_id='6', mask_num=mask_num, mode="validation", hasMasks=True,
                                      AutoChangeDim=False)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, pin_memory=False,
                                            num_workers=expConfig.DATASET_WORKERS)

    testset = DatasetLoader.OARDataset(TEST_PATH, expConfig, template_id='6', mask_num=None, mode="validation", hasMasks=False,
                                       AutoChangeDim=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, pin_memory=True,
                                             num_workers=expConfig.DATASET_WORKERS)

    expConfig.net = expConfig.NoNewReversible_affine_class_seg_highresolution()
    # expConfig.net = expConfig.NoNewReversible_affine_class()
    expConfig.net = expConfig.net.cuda()
    PRETRAINED_MODEL = True
    reg_model = "/media/root/01456da2-1f1d-4b67-810e-b9cd3341133d/2019_MICCAI_challenge_NPC/checkpoint/202007201646_MICCAI2015OAR_registration_fixed_64channels_reg_1_template2_resample/best_model.pt"
    # reg_model = "/media/root/01456da2-1f1d-4b67-810e-b9cd3341133d/2019_MICCAI_challenge_NPC/checkpoint/202007231924_MICCAI2015OAR_registration_fixed_64channels_2_221_reg_1_template2_resample/best_model.pt"
    if PRETRAINED_MODEL:
        checkpoint = torch.load(reg_model)
        model_dict = expConfig.net.state_dict()
        pretrained_dict = checkpoint["net_state_dict"]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        expConfig.net.load_state_dict(model_dict)

    total_num = sum(p.numel() for p in expConfig.net.parameters())
    trainiable_num = sum(p.numel() for p in expConfig.net.parameters() if p.requires_grad)
    print("total parameters:", total_num, " trainiable parameters:", trainiable_num)

    # expConfig.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, expConfig.net.parameters()), lr=expConfig.INITIAL_LR)
    expConfig.optimizer = RAdam(filter(lambda p: p.requires_grad, expConfig.net.parameters()), lr=expConfig.INITIAL_LR)
    expConfig.lr_sheudler = optim.lr_scheduler.CosineAnnealingLR(expConfig.optimizer, T_max=32)
    # expConfig.lr_sheudler = optim.lr_scheduler.OneCycleLR(expConfig.optimizer, max_lr=MAX_LR,
    #                                                       steps_per_epoch=len(trainloader), epochs=expConfig.EPOCHS)
    # expConfig.lr_sheudler = optim.lr_scheduler.MultiStepLR(expConfig.optimizer, [200], 0.2)

    seg = registrater.Registrater(expConfig, trainloader, valloader, testloader)

    ## save hyper parameters
    if expConfig.PREDICT == False:
        expConfig.EXCEL_SAVE_PATH = os.path.join(systemsetup.CHECKPOINT_BASE_PATH, expConfig.id, expConfig.EXCEL_ID)
        if os.path.exists(os.path.join(systemsetup.CHECKPOINT_BASE_PATH, expConfig.id)) == 0:
            os.makedirs(os.path.join(systemsetup.CHECKPOINT_BASE_PATH, expConfig.id))
        EXCEL_WORKBOOK = xlwt.Workbook()
        WORKSHEET = EXCEL_WORKBOOK.add_sheet('Sheet1')
        WORKSHEET.write(0, 0, 'Exp ID')
        WORKSHEET.write(0, 1, expConfig.id)
        WORKSHEET.write(1, 0, 'Exp Time')
        WORKSHEET.write(1, 1, exp_time)
        WORKSHEET.write(2, 0, 'Exp Task')
        WORKSHEET.write(2, 1, task_name)
        if exp_opt:
            WORKSHEET.write(3, 0, 'Exp Opt')
            WORKSHEET.write(3, 1, exp_opt)
        else:
            WORKSHEET.write(3, 0, 'Exp Opt')
            WORKSHEET.write(3, 1, 'None')
        WORKSHEET.write(4, 0, 'epochs')
        WORKSHEET.write(4, 1, str(expConfig.EPOCHS))
        WORKSHEET.write(5, 0, 'batchsize')
        WORKSHEET.write(5, 1, str(expConfig.BATCH_SIZE))
        WORKSHEET.write(5, 2, 'virtual batchsize')
        WORKSHEET.write(5, 3, str(expConfig.VIRTUAL_BATCHSIZE))
        WORKSHEET.write(6, 0, 'Total parameters')
        WORKSHEET.write(6, 1, str(total_num))
        WORKSHEET.write(6, 2, 'Trainiable parameters')
        WORKSHEET.write(6, 3, str(trainiable_num))
        WORKSHEET.write(7, 0, 'INITIAL_LR')
        WORKSHEET.write(7, 1, str(expConfig.INITIAL_LR))
        WORKSHEET.write(8, 0, 'MAX_LR')
        WORKSHEET.write(8, 1, str(MAX_LR))
        EXCEL_WORKBOOK.save(expConfig.EXCEL_SAVE_PATH)

    ## find best learning rate
    if AUTO_FIND_LR:
        seg.find_lr()

        # expConfig.optimizer = optim.AdamW(expConfig.net.parameters(), lr=expConfig.INITIAL_LR)
        # expConfig.lr_sheudler = optim.lr_scheduler.OneCycleLR(expConfig.optimizer, max_lr=MAX_LR,
        #                                                       steps_per_epoch=len(trainloader), epochs=expConfig.EPOCHS)
        # expConfig.lr_sheudler = optim.lr_scheduler.MultiStepLR(expConfig.optimizer, [100], 0.2)
    if AUTO_FIND_LR == False:
        if hasattr(expConfig, "VALIDATE_ALL") and expConfig.VALIDATE_ALL:
            seg.validateAllCheckpoints()
        elif hasattr(expConfig, "PREDICT") and expConfig.PREDICT:
            seg.makePredictions(target_class=mask_num)
        else:
            seg.train()

if __name__ == "__main__":
    main()
