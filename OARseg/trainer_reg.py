import torch
import RegDatasetLoader as DatasetLoader
import registrater as registrater
import systemsetup
import shutil
import os
import torch.optim as optim
import time
import experiments.Registration_Bin_OARs as expConfig
import xlwt,xlrd

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
    exp_opt = "registration_OneCycle_alltemplate2_scores_map_deform_affine_ratioresize"

    ## experiment name
    expConfig.id = exp_time + '_' + task_name + '_' + exp_opt if exp_opt else exp_time + '_' + task_name
    print("EXP ID:",expConfig.id)
    ## excel for logging
    expConfig.EXCEL_ID = 'training_info.xls'

    expConfig.PREDICT = False
    expConfig.RESTORE_ID = "202006220930_MICCAI2015OAR_registration_OneCycle_alltemplate2_moremorechannels_scores_map_affine_each_oar_dist"
    expConfig.MODEL_NAME = "last_model"

    AUTO_FIND_LR = False

    expConfig.INITIAL_LR = 1e-5
    MAX_LR = 1e-4
    expConfig.L2_REGULARIZER = 1e-5

    TRAIN_PATH = os.path.join(systemsetup.DATA_PATH, 'train_all_headonly_noresample_new')
    VAL_PATH = os.path.join(systemsetup.DATA_PATH, 'test_all_headonly_noresample_new')
    TEST_PATH = os.path.join(systemsetup.DATA_PATH, 'test_all_headonly_noresample_new')

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

    expConfig.net = expConfig.NoNewReversible_deform_affine()
    expConfig.optimizer = optim.Adam(expConfig.net.parameters(), lr=expConfig.INITIAL_LR)
    expConfig.lr_sheudler = optim.lr_scheduler.OneCycleLR(expConfig.optimizer, max_lr=MAX_LR,
                                                          steps_per_epoch=len(trainloader), epochs=expConfig.EPOCHS)
    # expConfig.lr_sheudler = optim.lr_scheduler.MultiStepLR(expConfig.optimizer, [200], 0.2)

    total_num = sum(p.numel() for p in expConfig.net.parameters())
    trainiable_num = sum(p.numel() for p in expConfig.net.parameters() if p.requires_grad)
    print("total parameters:", total_num, " trainiable parameters:", trainiable_num)

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
