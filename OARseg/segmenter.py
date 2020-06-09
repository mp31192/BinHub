import torch
import torch.nn as nn
import time
import NPCUtils
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import os
import sys
import dataProcessing.utils as utils
import systemsetup
import xlwt, xlrd
import xlutils.copy
import copy
from skimage.measure import regionprops
from torch.utils.checkpoint import checkpoint

# ################organs name###z,  y,  x#####################################
# organs_size = {'Brain Stem': [42, 54, 58],
#                'Eye ball Lens': [30, 44, 44],
#                'Optical Nerve Chiasm Pitutary': [26, 70, 78],
#                'Temporal Lobe': [44, 114, 74],
#                'Parotid glands': [44, 94, 60],
#                'Inner Middle ear': [38, 78, 64],
#                'Mandible T-M Joint': [56, 114, 88],
#                'Spinal cord': [86, 86, 40]}

################organs name########################################
organs_size = {'Brain Stem': [28, 20, 20, 28, 24, 12],
               'Optical Nerve': [42, 42, 4, 40, 42, 6],
               'Optical Chiasm': [42, 42, 4, 40, 42, 6],
               'Parotid glands': [44, 52, 18, 44, 32, 18],
               'Mandible': [76, 62, 16, 80, 80, 28],
               'Submandible glands': [18, 24, 10, 20, 24, 14]}

organs_size_class = {'Brain Stem': 0,
                     'Optical Nerve': 1,
                     'Optical Chiasm': 2,
                     'Parotid glands': 3,
                     'Mandible': 4,
                     'Submandible glands': 5}

organs_combine = {'Brain Stem':[1],
                  'Optical Nerve-L':[4],
                  'Optical Nerve-R':[5],
                  'Optical Chiasm':[2],
                  'Parotid glands-L':[6],'Parotid glands-R':[7],
                  'Mandible':[3], 'Submandible glands-L':[8], 'Submandible glands-R':[9]}

organs_channels = {'Brain Stem':[0],
                   'Optical Nerve-L':[1],
                   'Optical Nerve-R':[1],
                   'Optical Chiasm':[2],
                   'Parotid glands-L':[3],'Parotid glands-R':[3],
                   'Mandible':[4], 'Submandible glands-L':[5], 'Submandible glands-R':[5]}
organs_num = len(organs_combine)
organs_combine_name = list(organs_combine.keys())


class Segmenter:

    def __init__(self, expConfig, trainDataLoader, valDataLoader, testDataLoader):
        self.expConfig = expConfig
        self.trainDataLoader = trainDataLoader
        self.valDataLoader = valDataLoader
        self.testDataLoader = testDataLoader
        self.checkpointsBasePathLoad = systemsetup.CHECKPOINT_BASE_PATH
        self.checkpointsBasePathSave= systemsetup.CHECKPOINT_BASE_PATH
        self.predictionsBasePath = systemsetup.PREDICTIONS_BASE_PATH
        self.startFromEpoch = 1

        self.bestMeanDice = 0
        self.bestMeanDiceEpoch = 0

        self.movingAvg = 0
        self.bestMovingAvg = 0
        self.bestMovingAvgEpoch = 1e9
        self.EXPONENTIAL_MOVING_AVG_ALPHA = 0.95
        self.EARLY_STOPPING_AFTER_EPOCHS = 120


        # Run on GPU or CPU
        if torch.cuda.is_available():
            print("using cuda (", torch.cuda.device_count(), "device(s))")
            if torch.cuda.device_count() > 1:
                expConfig.net = nn.DataParallel(expConfig.net)
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            print("using cpu")

        # restore model if requested
        if hasattr(expConfig, "RESTORE_ID") and hasattr(expConfig, "MODEL_NAME") and expConfig.PREDICT:
            expConfig.net = expConfig.net.to(self.device)
            self.startFromEpoch = self.loadFromDisk(expConfig.RESTORE_ID, expConfig.MODEL_NAME)
            print("Loading checkpoint with id {} at epoch {}".format(expConfig.RESTORE_ID, expConfig.MODEL_NAME))

        # expConfig.net = expConfig.net.to(self.device)

    def validateAllCheckpoints(self):

        expConfig = self.expConfig

        print('==== VALIDATING ALL CHECKPOINTS ====')
        print(self.expConfig.EXPERIMENT_NAME)
        print("ID: {}".format(expConfig.id))
        print("RESTORE ID {}".format(expConfig.RESTORE_ID))
        print('====================================')

        for epoch in range(self.startFromEpoch, self.expConfig.EPOCHS):
            self.loadFromDisk(expConfig.RESTORE_ID, epoch)
            self.validate(epoch)

        #print best mean dice
        print("ID:", expConfig.id)
        print("Best mean dice: {:.4f} at epoch {}".format(self.bestMeanDice, self.bestMeanDiceEpoch))

    def makePredictions(self, target_class=9):
        # model is already loaded from disk by constructor

        expConfig = self.expConfig
        id = expConfig.RESTORE_ID

        print('============ PREDICTING ============')
        print("RESTORE ID {}".format(expConfig.RESTORE_ID))
        print("Model ", expConfig.MODEL_NAME)
        print('====================================')

        basePath = os.path.join(self.predictionsBasePath, id)
        if not os.path.exists(basePath):
            os.makedirs(basePath)

        expConfig.net.eval()

        with torch.no_grad():
            for i, data in enumerate(self.testDataLoader):
                inputs_list, patient_id = data
                print("processing {}".format(patient_id[0]))

                for ini in range(len(inputs_list)):
                    inputs_list[ini] = inputs_list[ini].to(self.device)
                inputs = inputs_list[0]
                local_result = inputs_list[1]

                outputs = torch.zeros_like(local_result)
                outputs_final_seg = torch.zeros_like(local_result)
                start_time = time.time()
                for mrf_i in range(0, 1):
                    for oar_i in range(0, organs_num):

                        if mrf_i == 0:
                            outputs_deform_new = copy.deepcopy(local_result)
                            outputs_mrf = torch.zeros_like(local_result)
                            organs_size_flag = organs_size
                        else:
                            outputs_deform_new = copy.deepcopy(outputs_final_seg)
                            outputs_mrf = copy.deepcopy(outputs_final_seg)
                            organs_size_flag = organs_size

                        organ_name_LR = organs_combine_name[oar_i]
                        organ_channels_list = organs_channels[organ_name_LR]

                        if '-L' in organ_name_LR:
                            organ_name = organ_name_LR.split('-L')[0]
                        elif '-R' in organ_name_LR:
                            organ_name = organ_name_LR.split('-R')[0]
                        else:
                            organ_name = organ_name_LR

                        crop_size = organs_size_flag[organ_name]
                        organ_id_combine = organs_combine[organ_name_LR]

                        for oic in range(len(organ_id_combine)):
                            organ_id = organ_id_combine[oic]
                            if oic == 0:
                                output_scores_map = outputs_deform_new[:, organ_id - 1:organ_id, :, :, :]
                                outputs_mrf_map = outputs_mrf[:, organ_id - 1:organ_id, :, :, :]
                            else:
                                output_scores_map = output_scores_map + outputs_deform_new[:, organ_id - 1:organ_id, :,
                                                                        :, :]
                                outputs_mrf_map = outputs_mrf_map + outputs_mrf[:, organ_id - 1:organ_id, :, :, :]

                        outputs_seg, mask_bbox = expConfig.net(inputs,
                                                               output_scores_map,
                                                               outputs_mrf_map,
                                                               crop_size)

                        for oic in range(len(organ_id_combine)):
                            organ_id = organ_id_combine[oic]
                            outchannel = organ_channels_list[oic]
                            outputs_final_seg[:, organ_id - 1, :, :, :] = 0
                            outputs_final_seg[:, organ_id - 1,
                            mask_bbox[2]:mask_bbox[5],
                            mask_bbox[1]:mask_bbox[4],
                            mask_bbox[0]:mask_bbox[3]] = outputs_seg[:, outchannel,
                                                         :mask_bbox[5] - mask_bbox[2],
                                                         :mask_bbox[4] - mask_bbox[1],
                                                         :mask_bbox[3] - mask_bbox[0]]
                            outputs[:, organ_id - 1,
                            mask_bbox[2]:mask_bbox[5],
                            mask_bbox[1]:mask_bbox[4],
                            mask_bbox[0]:mask_bbox[3]] = outputs_seg[:, outchannel,
                                                         :mask_bbox[5] - mask_bbox[2],
                                                         :mask_bbox[4] - mask_bbox[1],
                                                         :mask_bbox[3] - mask_bbox[0]]

                print("Spend Time:", time.time() - start_time)
                fullsize = outputs

                #binarize output
                wt = fullsize.chunk(target_class, dim=1)
                wt = list(wt)
                # wt = wt[0]
                wt_num = len(wt)
                s = fullsize.shape
                for wn in range(wt_num):
                    wt[wn] = (wt[wn] > 0.5).view(s[2], s[3], s[4])

                result = fullsize.new_zeros((s[2], s[3], s[4]), dtype=torch.uint8)
                for wn in range(wt_num):
                    result[wt[wn]] = wn+1

                npResult = result[:, :, :].cpu().numpy()
                npResult = np.transpose(npResult, [1,0,2])
                path = os.path.join(basePath, "{}_result.nii.gz".format(patient_id[0]))
                utils.save_nii(path, npResult, None, None)

        print("Done :)")

    def find_lr(self):
        expConfig = self.expConfig

        print('======= FINDING LEARNING RATE =======')
        print("ID: {}".format(expConfig.id))
        print('=====================================')

        epoch = 1
        while epoch < 10 and epoch <= self.bestMovingAvgEpoch + self.EARLY_STOPPING_AFTER_EPOCHS:

            expConfig.net = expConfig.NoNewReversible_multiview_with_3d_mrf(Final_output=6,
                                                                            Input_Channels=1,
                                                                            kernel_size=(3, 3, 3),
                                                                            kernel_size_2=(3, 3, 1))
            expConfig.net = expConfig.net.to(self.device)
            expConfig.INITIAL_LR = 1e-9 * 10 ** (epoch - 1)
            expConfig.optimizer = optim.AdamW(expConfig.net.parameters(), lr=expConfig.INITIAL_LR)
            expConfig.optimizer.zero_grad()

            running_loss = 0.0
            epoch_running_loss = 0.0
            startTime = time.time()

            # set net up training
            self.expConfig.net.train()

            for i, data in enumerate(self.trainDataLoader):

                # load data
                inputs, pid, labels = data
                for ini in range(len(inputs)):
                    inputs[ini] = inputs[ini].to(self.device)
                for lai in range(len(labels)):
                    labels[lai] = labels[lai].to(self.device)

                ori_inputs, ori_labels = inputs[0], labels[0]
                ori_local_result = inputs[1]

                total_loss = 0

                outputs_final_seg = torch.zeros_like(ori_local_result)
                for mrf_i in range(0, 1):
                    organ_num_count = 0
                    loss_n = 0
                    for oar_i in range(0, organs_num):

                        inputs = copy.deepcopy(ori_inputs)
                        labels = copy.deepcopy(ori_labels)
                        local_result = copy.deepcopy(ori_local_result)
                        if mrf_i == 0 or epoch < 10:
                            outputs_deform_new = copy.deepcopy(labels)
                            if mrf_i == 0:
                                outputs_mrf = torch.zeros_like(labels)
                                organs_size_flag = organs_size
                            else:
                                outputs_mrf = torch.from_numpy(outputs_final_seg.detach().cpu().numpy()).to(self.device)
                                organs_size_flag = organs_size
                        else:
                            outputs_deform_new = torch.from_numpy(outputs_final_seg.detach().cpu().numpy()).to(
                                self.device)
                            outputs_mrf = torch.from_numpy(outputs_final_seg.detach().cpu().numpy()).to(self.device)
                            if mrf_i == 0:
                                organs_size_flag = organs_size
                            else:
                                organs_size_flag = organs_size

                        labels_select = torch.zeros_like(labels)
                        outputs = torch.zeros_like(labels)

                        organ_name_LR = organs_combine_name[oar_i]
                        organ_channels_list = organs_channels[organ_name_LR]
                        # print(organ_name_LR)
                        if '-L' in organ_name_LR:
                            organ_name = organ_name_LR.split('-L')[0]
                        elif '-R' in organ_name_LR:
                            organ_name = organ_name_LR.split('-R')[0]
                        else:
                            organ_name = organ_name_LR

                        class_id = organs_size_class[organ_name]
                        labels_class = torch.tensor([class_id]).to(self.device)
                        crop_size = organs_size_flag[organ_name]
                        organ_id_combine = organs_combine[organ_name_LR]

                        for oic in range(len(organ_id_combine)):
                            organ_id = organ_id_combine[oic]
                            if oic == 0:
                                output_scores_map = outputs_deform_new[:, organ_id - 1:organ_id, :, :, :]
                                outputs_mrf_map = outputs_mrf[:, organ_id - 1:organ_id, :, :, :]
                                labels_flag = labels[:, organ_id - 1:organ_id, :, :, :]
                            else:
                                output_scores_map = output_scores_map + outputs_deform_new[:, organ_id - 1:organ_id, :,
                                                                        :, :]
                                outputs_mrf_map = outputs_mrf_map + outputs_mrf[:, organ_id - 1:organ_id, :, :, :]
                                labels_flag = labels_flag + labels[:, organ_id - 1:organ_id, :, :, :]


                        if torch.sum(labels_flag).item() == 0:
                            continue

                        outputs_seg, mask_bbox = expConfig.net(inputs,
                                                               output_scores_map,
                                                               outputs_mrf_map,
                                                               crop_size)

                        for oic in range(len(organ_id_combine)):
                            organ_id = organ_id_combine[oic]
                            outchannel = organ_channels_list[oic]
                            outputs_final_seg[:, organ_id - 1, :, :, :] = 0
                            outputs_final_seg[:, organ_id - 1,
                            mask_bbox[2]:mask_bbox[5],
                            mask_bbox[1]:mask_bbox[4],
                            mask_bbox[0]:mask_bbox[3]] = outputs_seg[:, outchannel,
                                                         :mask_bbox[5] - mask_bbox[2],
                                                         :mask_bbox[4] - mask_bbox[1],
                                                         :mask_bbox[3] - mask_bbox[0]]
                            outputs[:, organ_id - 1,
                            mask_bbox[2]:mask_bbox[5],
                            mask_bbox[1]:mask_bbox[4],
                            mask_bbox[0]:mask_bbox[3]] = outputs_seg[:, outchannel,
                                                         :mask_bbox[5] - mask_bbox[2],
                                                         :mask_bbox[4] - mask_bbox[1],
                                                         :mask_bbox[3] - mask_bbox[0]]

                            labels_select[:, organ_id - 1,
                            mask_bbox[2]:mask_bbox[5],
                            mask_bbox[1]:mask_bbox[4],
                            mask_bbox[0]:mask_bbox[3]] = labels[:, organ_id - 1,
                                                         mask_bbox[2]:mask_bbox[5],
                                                         mask_bbox[1]:mask_bbox[4],
                                                         mask_bbox[0]:mask_bbox[3]]

                            # outputs_crop_val[:, organ_id - 1, mask_bbox[2]:mask_bbox[5], mask_bbox[1]:mask_bbox[4],
                            #                  mask_bbox[0]:mask_bbox[3]] = 0

                        loss_dice, loss2_list = expConfig.loss(outputs, labels_select)

                        print(mrf_i, " Seg Loss:", loss_dice.item())

                        loss = loss_dice

                        loss_n = loss_n + loss.item()

                        loss = loss / organs_num

                        loss.backward()


                        del inputs, labels, \
                            outputs, outputs_seg, \
                            labels_class, labels_select, outputs_deform_new, \
                            outputs_mrf, outputs_mrf_map, output_scores_map, \
                            mask_bbox

                        organ_num_count += 1

                    # update params
                    expConfig.optimizer.step()
                    expConfig.optimizer.zero_grad()

                    total_loss = total_loss + loss_n / organ_num_count

                torch.cuda.empty_cache()

                del ori_inputs, ori_labels, outputs_final_seg, class_id, crop_size, data

                loss = loss / (mrf_i + 1)

                running_loss += total_loss
                epoch_running_loss += total_loss
                del loss
                if expConfig.LOG_EVERY_K_ITERATIONS > 0:
                    if i % expConfig.LOG_EVERY_K_ITERATIONS == (expConfig.LOG_EVERY_K_ITERATIONS - 1):
                        print('[%d, %5d] loss: %.3f' % (epoch, i + 1, running_loss / expConfig.LOG_EVERY_K_ITERATIONS))
                        running_loss = 0.0

            # logging at end of epoch
            if expConfig.LOG_EPOCH_TIME:
                print("Time for epoch: {:.2f}s".format(time.time() - startTime))
                self.movingAvg = copy.deepcopy(epoch_running_loss / (i + 1))
                print("Loss for epoch: ", epoch_running_loss / (i + 1))

            if expConfig.LOG_LR_EVERY_EPOCH:
                for param_group in expConfig.optimizer.param_groups:
                    lr_now = param_group['lr']
                    print("Current lr: {:.6f}".format(param_group['lr']))

            # validation at end of epoch
            if epoch % expConfig.VALIDATE_EVERY_K_EPOCHS == expConfig.VALIDATE_EVERY_K_EPOCHS - 1:
                if os.path.exists(expConfig.EXCEL_SAVE_PATH):
                    EXCEL_WORKBOOK = xlrd.open_workbook(expConfig.EXCEL_SAVE_PATH)
                    WORKSHEET = EXCEL_WORKBOOK.sheet_by_name('Sheet1')
                    # cell_11 = WORKSHEET.cell(0, 7).value

                    if WORKSHEET.ncols < 7:
                        EXCEL_WORKBOOK = xlutils.copy.copy(EXCEL_WORKBOOK)
                        WORKSHEET = EXCEL_WORKBOOK.get_sheet('Sheet1')
                        WORKSHEET.write(0, 7, 'lr')
                        WORKSHEET.write(0, 8, 'loss')
                        WORKSHEET.write(epoch, 7, str(lr_now))
                        WORKSHEET.write(epoch, 8, str(self.movingAvg))
                        EXCEL_WORKBOOK.save(expConfig.EXCEL_SAVE_PATH)
                    else:
                        EXCEL_WORKBOOK = xlutils.copy.copy(EXCEL_WORKBOOK)
                        WORKSHEET = EXCEL_WORKBOOK.get_sheet('Sheet1')
                        WORKSHEET.write(epoch, 7, str(lr_now))
                        WORKSHEET.write(epoch, 8, str(self.movingAvg))
                        EXCEL_WORKBOOK.save(expConfig.EXCEL_SAVE_PATH)
                else:
                    print("\033[31m" + "Excel is not exist while Finding Learning Rate!")
                    sys.exit(1)

            epoch = epoch + 1

    def train(self):
        expConfig = self.expConfig
        expConfig.net.to(self.device)
        print('============== TRAINING ==============')
        print("ID: {}".format(expConfig.id))
        print('======================================')

        epoch = self.startFromEpoch
        while epoch < expConfig.EPOCHS and epoch <= self.bestMovingAvgEpoch + self.EARLY_STOPPING_AFTER_EPOCHS:
            expConfig.net.train()

            expConfig.net = expConfig.net.to(self.device)
            expConfig.optimizer.zero_grad()

            running_loss = 0.0
            epoch_running_loss = 0.0
            startTime = time.time()

            for i, data in enumerate(self.trainDataLoader):

                # load data
                inputs_list, pid, labels_list = data
                for ini in range(len(inputs_list)):
                    inputs_list[ini] = inputs_list[ini].to(self.device)
                for lai in range(len(labels_list)):
                    labels_list[lai] = labels_list[lai].to(self.device)

                ori_inputs, ori_labels = inputs_list[0], labels_list[0]
                ori_local_result = inputs_list[1]

                total_loss = 0

                outputs_final_seg = torch.zeros_like(ori_local_result)
                for mrf_i in range(0, 2):
                    organ_num_count = 0
                    loss_n = 0
                    for oar_i in range(0, organs_num):

                        inputs = copy.deepcopy(ori_inputs)
                        labels = copy.deepcopy(ori_labels)
                        local_result = copy.deepcopy(ori_local_result)
                        if mrf_i == 0 or epoch < 10:
                            outputs_deform_new = copy.deepcopy(local_result)
                            if mrf_i == 0:
                                outputs_mrf = torch.zeros_like(local_result)
                                organs_size_flag = organs_size
                            else:
                                outputs_mrf = torch.from_numpy(outputs_final_seg.detach().cpu().numpy()).to(self.device)
                                organs_size_flag = organs_size
                        else:
                            outputs_deform_new = torch.from_numpy(outputs_final_seg.detach().cpu().numpy()).to(
                                self.device)
                            outputs_mrf = torch.from_numpy(outputs_final_seg.detach().cpu().numpy()).to(self.device)
                            if mrf_i == 0:
                                organs_size_flag = organs_size
                            else:
                                organs_size_flag = organs_size

                        labels_select = torch.zeros_like(local_result)
                        outputs = torch.zeros_like(local_result)

                        organ_name_LR = organs_combine_name[oar_i]
                        organ_channels_list = organs_channels[organ_name_LR]
                        # print(organ_name_LR)
                        if '-L' in organ_name_LR:
                            organ_name = organ_name_LR.split('-L')[0]
                        elif '-R' in organ_name_LR:
                            organ_name = organ_name_LR.split('-R')[0]
                        else:
                            organ_name = organ_name_LR

                        class_id = organs_size_class[organ_name]
                        labels_class = torch.tensor([class_id]).to(self.device)
                        crop_size = organs_size_flag[organ_name]
                        organ_id_combine = organs_combine[organ_name_LR]

                        for oic in range(len(organ_id_combine)):
                            organ_id = organ_id_combine[oic]
                            if oic == 0:
                                output_scores_map = outputs_deform_new[:, organ_id - 1:organ_id, :, :, :]
                                outputs_mrf_map = outputs_mrf[:, organ_id - 1:organ_id, :, :, :]
                                labels_flag = labels[:, organ_id - 1:organ_id, :, :, :]
                            else:
                                output_scores_map = output_scores_map + outputs_deform_new[:, organ_id - 1:organ_id, :,
                                                                        :, :]
                                outputs_mrf_map = outputs_mrf_map + outputs_mrf[:, organ_id - 1:organ_id, :, :, :]
                                labels_flag = labels_flag + labels[:, organ_id - 1:organ_id, :, :, :]

                        if torch.sum(labels_flag).item() == 0:
                            continue

                        outputs_seg, mask_bbox = expConfig.net(inputs,
                                                               output_scores_map,
                                                               outputs_mrf_map,
                                                               crop_size)

                        for oic in range(len(organ_id_combine)):
                            organ_id = organ_id_combine[oic]
                            outchannel = organ_channels_list[oic]
                            outputs_final_seg[:, organ_id - 1, :, :, :] = 0
                            outputs_final_seg[:, organ_id - 1,
                            mask_bbox[2]:mask_bbox[5],
                            mask_bbox[1]:mask_bbox[4],
                            mask_bbox[0]:mask_bbox[3]] = outputs_seg[:, outchannel,
                                                         :mask_bbox[5] - mask_bbox[2],
                                                         :mask_bbox[4] - mask_bbox[1],
                                                         :mask_bbox[3] - mask_bbox[0]]
                            outputs[:, organ_id - 1,
                            mask_bbox[2]:mask_bbox[5],
                            mask_bbox[1]:mask_bbox[4],
                            mask_bbox[0]:mask_bbox[3]] = outputs_seg[:, outchannel,
                                                         :mask_bbox[5] - mask_bbox[2],
                                                         :mask_bbox[4] - mask_bbox[1],
                                                         :mask_bbox[3] - mask_bbox[0]]

                            labels_select[:, organ_id - 1,
                            mask_bbox[2]:mask_bbox[5],
                            mask_bbox[1]:mask_bbox[4],
                            mask_bbox[0]:mask_bbox[3]] = labels[:, organ_id - 1,
                                                         mask_bbox[2]:mask_bbox[5],
                                                         mask_bbox[1]:mask_bbox[4],
                                                         mask_bbox[0]:mask_bbox[3]]

                            # outputs_crop_val[:, organ_id - 1, mask_bbox[2]:mask_bbox[5], mask_bbox[1]:mask_bbox[4],
                            #                  mask_bbox[0]:mask_bbox[3]] = 0

                        loss_dice = expConfig.loss(outputs, labels_select)

                        print(mrf_i, " Seg Loss:", loss_dice.item())

                        loss = loss_dice

                        loss_n = loss_n + loss.item()

                        loss = loss / organs_num

                        loss.backward()

                        del inputs, labels, \
                            outputs, outputs_seg, \
                            labels_class, labels_select, outputs_deform_new, \
                            outputs_mrf, outputs_mrf_map, output_scores_map, \
                            mask_bbox, local_result, loss

                        organ_num_count += 1

                    for param_group in expConfig.optimizer.param_groups:
                        print("Current lr: {:.6f}".format(param_group['lr']))
                    # update params
                    expConfig.optimizer.step()
                    expConfig.optimizer.zero_grad()

                    ## take lr sheudler step
                    if hasattr(expConfig, "lr_sheudler"):
                        if isinstance(expConfig.lr_sheudler, optim.lr_scheduler.OneCycleLR) and mrf_i == 1:
                            expConfig.lr_sheudler.step()

                    total_loss = total_loss + loss_n / organ_num_count

                    torch.cuda.empty_cache()

                del ori_inputs, ori_labels, ori_local_result, outputs_final_seg, class_id, crop_size, data, \
                    inputs_list, labels_list

                total_loss = total_loss / (mrf_i + 1)

                running_loss += total_loss
                epoch_running_loss += total_loss

                del total_loss
                if expConfig.LOG_EVERY_K_ITERATIONS > 0:
                    if i % expConfig.LOG_EVERY_K_ITERATIONS == (expConfig.LOG_EVERY_K_ITERATIONS - 1):
                        print('[%d, %5d] loss: %.3f' % (epoch, i + 1, running_loss / expConfig.LOG_EVERY_K_ITERATIONS))
                        if expConfig.LOG_MEMORY_EVERY_K_ITERATIONS: self.logMemoryUsage()
                        running_loss = 0.0

            #logging at end of epoch
            if expConfig.LOG_EPOCH_TIME:
                print("Time for epoch: {:.2f}s".format(time.time() - startTime))
                print("Loss for epoch: ", epoch_running_loss/(i + 1))

            if expConfig.LOG_LR_EVERY_EPOCH:
                for param_group in expConfig.optimizer.param_groups:
                    print("Current lr: {:.6f}".format(param_group['lr']))

            #validation at end of epoch
            if epoch % expConfig.VALIDATE_EVERY_K_EPOCHS == expConfig.VALIDATE_EVERY_K_EPOCHS - 1:
                if os.path.exists(expConfig.EXCEL_SAVE_PATH):
                    EXCEL_WORKBOOK = xlrd.open_workbook(expConfig.EXCEL_SAVE_PATH)
                    sheets_num = len(EXCEL_WORKBOOK.sheets())
                    EXCEL_WORKBOOK = xlutils.copy.copy(EXCEL_WORKBOOK)

                    if sheets_num == 1:
                        WORKSHEET = EXCEL_WORKBOOK.add_sheet('Sheet2')
                        WORKSHEET.write(0, 0, 'Epoch')
                        WORKSHEET.write(0, 1, 'VAL DICE')
                        WORKSHEET.write(0, 2, 'TRAIN DICE')
                        WORKSHEET.write(0, 3, 'TRAIN LOSS')
                        WORKSHEET.write(0, 4, 'LEARNING RATE')
                    else:
                        WORKSHEET = EXCEL_WORKBOOK.get_sheet('Sheet2')

                    WORKSHEET.write(epoch, 0, str(epoch))
                    WORKSHEET.write(epoch, 3, str(epoch_running_loss / (i + 1)))
                    WORKSHEET.write(epoch, 4, str(param_group['lr']))
                    EXCEL_WORKBOOK.save(expConfig.EXCEL_SAVE_PATH)

                self.validate(epoch)

            #take lr sheudler step
            if hasattr(expConfig, "lr_sheudler"):
                if isinstance(expConfig.lr_sheudler, optim.lr_scheduler.ReduceLROnPlateau):
                    expConfig.lr_sheudler.step(self.movingAvg)
                elif isinstance(expConfig.lr_sheudler, optim.lr_scheduler.MultiStepLR):
                    expConfig.lr_sheudler.step()

            #save model
            if expConfig.SAVE_CHECKPOINTS:
                self.saveToDisk(epoch)

            epoch = epoch + 1

        #print best mean dice
        print("ID:",expConfig.id)
        print("Best mean dice: {:.4f} at epoch {}".format(self.bestMeanDice, self.bestMeanDiceEpoch))

    def validate(self, epoch, target_class = 9):

        #set net up for inference
        expConfig = self.expConfig
        expConfig.net.eval()

        startTime = time.time()
        with torch.no_grad():
            diceWT = []
            for tc in range(target_class):
                diceWT.append([])
            sensWT = []
            for tc in range(target_class):
                sensWT.append([])
            specWT = []
            for tc in range(target_class):
                specWT.append([])
            hdWT = []
            for tc in range(target_class):
                hdWT.append([])

            for i, data in enumerate(self.valDataLoader):

                # feed inputs through neural net
                inputs_list, _, labels_list = data
                for ini in range(len(inputs_list)):
                    inputs_list[ini] = inputs_list[ini].to(self.device)
                for lai in range(len(labels_list)):
                    labels_list[lai] = labels_list[lai].to(self.device)
                inputs, labels = inputs_list[0], labels_list[0]
                local_result = inputs_list[1]

                outputs_deform = copy.deepcopy(local_result)
                outputs = torch.zeros_like(outputs_deform)
                outputs_final_seg = torch.zeros_like(local_result)
                for mrf_i in range(0, 2):
                    for oar_i in range(0, organs_num):

                        if mrf_i == 0:
                            outputs_deform_new = copy.deepcopy(local_result)
                            outputs_mrf = torch.zeros_like(local_result)
                            organs_size_flag = organs_size
                        else:
                            outputs_deform_new = copy.deepcopy(outputs_final_seg)
                            outputs_mrf = copy.deepcopy(outputs_final_seg)
                            organs_size_flag = organs_size

                        organ_name_LR = organs_combine_name[oar_i]
                        organ_channels_list = organs_channels[organ_name_LR]
                        # print(organ_name_LR)
                        if '-L' in organ_name_LR:
                            organ_name = organ_name_LR.split('-L')[0]
                        elif '-R' in organ_name_LR:
                            organ_name = organ_name_LR.split('-R')[0]
                        else:
                            organ_name = organ_name_LR

                        crop_size = organs_size_flag[organ_name]
                        organ_id_combine = organs_combine[organ_name_LR]

                        for oic in range(len(organ_id_combine)):
                            organ_id = organ_id_combine[oic]
                            if oic == 0:
                                output_scores_map = outputs_deform_new[:, organ_id - 1:organ_id, :, :, :]
                                outputs_mrf_map = outputs_mrf[:, organ_id - 1:organ_id, :, :, :]
                            else:
                                output_scores_map = output_scores_map + outputs_deform_new[:, organ_id - 1:organ_id, :,
                                                                        :, :]
                                outputs_mrf_map = outputs_mrf_map + outputs_mrf[:, organ_id - 1:organ_id, :, :, :]

                        outputs_seg, mask_bbox = expConfig.net(inputs,
                                                               output_scores_map,
                                                               outputs_mrf_map,
                                                               crop_size)

                        for oic in range(len(organ_id_combine)):
                            organ_id = organ_id_combine[oic]
                            outchannel = organ_channels_list[oic]
                            outputs_final_seg[:, organ_id - 1, :, :, :] = 0
                            outputs_final_seg[:, organ_id - 1,
                            mask_bbox[2]:mask_bbox[5],
                            mask_bbox[1]:mask_bbox[4],
                            mask_bbox[0]:mask_bbox[3]] = outputs_seg[:, outchannel,
                                                         :mask_bbox[5] - mask_bbox[2],
                                                         :mask_bbox[4] - mask_bbox[1],
                                                         :mask_bbox[3] - mask_bbox[0]]
                            outputs[:, organ_id - 1,
                            mask_bbox[2]:mask_bbox[5],
                            mask_bbox[1]:mask_bbox[4],
                            mask_bbox[0]:mask_bbox[3]] = outputs_seg[:, outchannel,
                                                         :mask_bbox[5] - mask_bbox[2],
                                                         :mask_bbox[4] - mask_bbox[1],
                                                         :mask_bbox[3] - mask_bbox[0]]

                del outputs_deform_new, \
                    outputs_mrf, outputs_mrf_map, output_scores_map, \
                    mask_bbox, local_result
                del outputs_final_seg, crop_size, data, \
                    inputs_list, labels_list

                #separate outputs channelwise
                wt = outputs.chunk(target_class, dim=1)
                wt = list(wt)
                wt_num = len(wt)
                s = wt[0].shape
                for wn in range(wt_num):
                    wt[wn] = wt[wn].view(s[0], s[2], s[3], s[4])


                wtMask = labels.chunk(target_class, dim=1)
                wtMask = list(wtMask)
                s = wtMask[0].shape
                for wn in range(wt_num):
                    wtMask[wn] = wtMask[wn].view(s[0], s[2], s[3], s[4])

                #get dice metrics
                for wn in range(wt_num):
                    diceWT[wn].append(NPCUtils.dice(wt[wn], wtMask[wn]))

                #get sensitivity metrics
                for wn in range(wt_num):
                    sensWT[wn].append(NPCUtils.sensitivity(wt[wn], wtMask[wn]))

                #get specificity metrics
                for wn in range(wt_num):
                    specWT[wn].append(NPCUtils.specificity(wt[wn], wtMask[wn]))

        #calculate mean dice scores
        meanDiceWT = []
        meanDice = 0
        for wn in range(wt_num):
            meanDiceWT.append(np.mean(diceWT[wn]))
            # meanDice = meanDice + meanDiceWT[wn] * OARs_weight[str(wn+1)]
        # meanDice = meanDice / weight_sum
        meanDice = np.mean([meanDiceWT])

        if (meanDice > self.bestMeanDice):
            self.bestMeanDice = meanDice
            self.bestMeanDiceEpoch = epoch

        #update moving avg
        self._updateMovingAvg(meanDice, epoch)

        EXCEL_WORKBOOK = xlrd.open_workbook(expConfig.EXCEL_SAVE_PATH)
        EXCEL_WORKBOOK = xlutils.copy.copy(EXCEL_WORKBOOK)
        WORKSHEET = EXCEL_WORKBOOK.get_sheet('Sheet2')
        WORKSHEET.write(epoch, 0, str(epoch))
        WORKSHEET.write(epoch, 1, str(meanDice))
        EXCEL_WORKBOOK.save(expConfig.EXCEL_SAVE_PATH)

        #print metrics
        print("------ Validation epoch {} ------".format(epoch))
        for wn in range(wt_num):
            print("Dice        WT: {:.4f} Mean: {:.4f} MovingAvg: {:.4f}".format(meanDiceWT[wn], meanDice, self.movingAvg))
            print("Sensitivity WT: {:.4f}".format(np.mean(sensWT[wn])))
            print("Specificity WT: {:.4f}".format(np.mean(specWT[wn])))

        #log validation time
        if expConfig.LOG_VALIDATION_TIME:
            print("Time for validation: {:.2f}s".format(time.time() - startTime))
        print("--------------------------------")

    def logMemoryUsage(self, additionalString=""):
        if torch.cuda.is_available():
            print(additionalString + "Memory {:.0f}Mb max, {:.0f}Mb current".format(
                torch.cuda.max_memory_allocated() / 1024 / 1024, torch.cuda.memory_allocated() / 1024 / 1024))


    def saveToDisk(self, epoch):

        #gather things to save
        saveDict = {"net_state_dict": self.expConfig.net.state_dict(),
                    "optimizer_state_dict": self.expConfig.optimizer.state_dict(),
                    "epoch": epoch,
                    "bestMeanDice": self.bestMeanDice,
                    "bestMeanDiceEpoch": self.bestMeanDiceEpoch,
                    "movingAvg": self.movingAvg,
                    "bestMovingAvgEpoch": self.bestMovingAvgEpoch,
                    "bestMovingAvg": self.bestMovingAvg}
        if hasattr(self.expConfig, "lr_sheudler"):
            saveDict["lr_sheudler_state_dict"] = self.expConfig.lr_sheudler.state_dict()

        #save dict
        basePath = self.checkpointsBasePathSave + "{}".format(self.expConfig.id)
        if not os.path.exists(basePath):
            os.makedirs(basePath)
        if self.bestMeanDiceEpoch == epoch:
            path = basePath + "/best_model.pt"
            torch.save(saveDict, path)

        path = basePath + "/last_model.pt"
        torch.save(saveDict, path)

    def loadFromDisk(self, id, model_name):
        path = self._getCheckpointPathLoad(id, model_name)
        checkpoint = torch.load(path)
        self.expConfig.net.load_state_dict(checkpoint["net_state_dict"])

        #load optimizer: hack necessary because load_state_dict has bugs (See https://github.com/pytorch/pytorch/issues/2830#issuecomment-336194949)
        self.expConfig.optimizer.load_state_dict(checkpoint["optimizer_state_dict_seg"])
        for state in self.expConfig.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    if torch.cuda.is_available():
                        state[k] = v.cuda()
                    else:
                        state[k] = v

        if "lr_sheudler_state_dict" in checkpoint:
            self.expConfig.lr_sheudler.load_state_dict(checkpoint["lr_sheudler_state_dict"])
            #Hack lr sheudle
            #self.expConfig.lr_sheudler.milestones = [250, 400, 550]

        #load best epoch score (if available)
        if "bestMeanDice" in checkpoint:
            self.bestMeanDice = checkpoint["bestMeanDice"]
            self.bestMeanDiceEpoch = checkpoint["bestMeanDiceEpoch"]

        #load moving avg if available
        if "movingAvg" in checkpoint:
            self.movingAvg = checkpoint["movingAvg"]

        #load best moving avg epoch if available
        if "bestMovingAvgEpoch" in checkpoint:
            self.bestMovingAvgEpoch = checkpoint["bestMovingAvgEpoch"]
        if "bestMovingAvg" in checkpoint:
            self.bestMovingAvg = checkpoint["bestMovingAvg"]

        return checkpoint["epoch"]

    def _getCheckpointPathLoad(self, id, model_name):
        return os.path.join(self.checkpointsBasePathLoad, id, model_name+'.pt')

    def _updateMovingAvg(self, validationMean, epoch):
        if self.movingAvg == 0:
            self.movingAvg = validationMean
        else:
            alpha = self.EXPONENTIAL_MOVING_AVG_ALPHA
            self.movingAvg = self.movingAvg * alpha + validationMean * (1 - alpha)

        if self.bestMovingAvg < self.movingAvg:
            self.bestMovingAvg = self.movingAvg
            self.bestMovingAvgEpoch = epoch

    def new_crop(self, outputs):
        outputs_cpu = outputs.detach().cpu().numpy()
        # b, c, x, y, z = outputs_cpu.shape
        organs_size_new = {}

        for i in range(len(organs_combine)):
            organs_name = organs_combine_name[i]
            organs_combine_list = organs_combine[organs_name]
            organs_name = organs_name.split('-')[0]
            for oc in organs_combine_list:
                if 'outputs_one' not in dir():
                    outputs_one = outputs_cpu[0, oc - 1, :, :, :]
                else:
                    outputs_one_a = outputs_cpu[0, oc - 1, :, :, :]
                    outputs_one = outputs_one + outputs_one_a

            outputs_one = outputs_one / len(organs_combine_list)
            outputs_one[outputs_one >= 0.5] = 1
            outputs_one[outputs_one < 0.5] = 0
            # outputs_one[outputs_one > 1] = 1
            outputs_one = outputs_one.astype('int')
            outputs_props = regionprops(outputs_one)

            if len(outputs_props) > 0:
                outputs_props = outputs_props[0]
                bbox = outputs_props['bbox']
                center_point = outputs_props['centroid']
                max_x = int(bbox[3] - center_point[0])
                min_x = int(center_point[0] - bbox[0])
                max_y = int(bbox[4] - center_point[1])
                min_y = int(center_point[1] - bbox[1])
                max_z = int(bbox[5] - center_point[2])
                min_z = int(center_point[2] - bbox[2])
                organ_size_new = [min_x, min_y, min_z, max_x, max_y, max_z]
                organ_size_old = organs_size[organs_name]
                for mi in range(0, 2):
                    # if mi != 2:
                    organ_size_new[mi] = np.minimum(organ_size_new[mi] + (organ_size_new[mi] % 2) + 6
                                                    , organ_size_old[mi])
                    organ_size_new[mi + 3] = np.minimum(organ_size_new[mi + 3] + (organ_size_new[mi + 3] % 2) + 6
                                                        , organ_size_old[mi + 3])
                mi = 2
                organ_size_new[mi] = np.minimum(organ_size_new[mi] + (organ_size_new[mi] % 2) + 4
                                                , organ_size_old[mi])
                organ_size_new[mi + 3] = np.minimum(organ_size_new[mi + 3] + (organ_size_new[mi + 3] % 2 + 4)
                                                    , organ_size_old[mi + 3])
                    # else:
                    #     organ_size_new[mi] = np.minimum(organ_size_new[mi] + (organ_size_new[mi] % 2)
                    #                                     , organ_size_old[mi])
                    #     organ_size_new[mi + 3] = np.minimum(organ_size_new[mi + 3] + (organ_size_new[mi + 3] % 2)
                    #                                         , organ_size_old[mi + 3])
            elif len(outputs_props) == 0:
                organ_size_new = organs_size[organs_name]
            organs_size_new.update({organs_name: organ_size_new})
        del outputs_cpu, outputs_one, outputs_one_a, organ_size_old, organ_size_new, outputs, outputs_props
        return organs_size_new
