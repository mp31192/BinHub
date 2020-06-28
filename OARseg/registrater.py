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
import random
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
class Registrater:

    def __init__(self, expConfig, trainDataLoader, valDataLoader, testDataLoader):
        self.expConfig = expConfig
        self.trainDataLoader = trainDataLoader
        self.valDataLoader = valDataLoader
        self.testDataLoader = testDataLoader
        self.checkpointsBasePathLoad = systemsetup.CHECKPOINT_BASE_PATH
        self.checkpointsBasePathSave= systemsetup.CHECKPOINT_BASE_PATH
        self.predictionsBasePath = systemsetup.PREDICTIONS_BASE_PATH
        self.startFromEpoch = 1
        self.mode = "train"

        self.bestMeanDice = 0
        self.bestMeanDiceEpoch = 0

        self.epoch_now = 0
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
        self.mode = "eval"

        with torch.no_grad():
            for i, data in enumerate(self.testDataLoader):
                start_time = time.time()

                inputs_list, pid = data

                inputs, inputs_atlas, inputs_atlas_mask = \
                    inputs_list[0], inputs_list[1], inputs_list[2]

                inputs_shape = inputs.shape

                inputs = F.interpolate(inputs, size=(224, 160, 96), mode='trilinear', align_corners=True)
                inputs_atlas = F.interpolate(inputs_atlas, size=(224, 160, 96), mode='trilinear', align_corners=True)
                inputs_atlas_mask = F.interpolate(inputs_atlas_mask, size=(224, 160, 96), mode='nearest')

                inputs, inputs_atlas, inputs_atlas_mask = \
                    inputs.to(self.device), inputs_atlas.to(self.device), inputs_atlas_mask.to(self.device)

                outputs_atlas_mask_affine = \
                    expConfig.net(inputs, inputs_atlas, inputs_atlas_mask)

                # outputs_atlas = F.interpolate(outputs_atlas, size=(inputs_shape[2], inputs_shape[3], inputs_shape[4]),
                #                               mode='trilinear', align_corners=True)
                # outputs_atlas_mask = F.interpolate(outputs_atlas_mask, size=(inputs_shape[2], inputs_shape[3], inputs_shape[4]),
                #                                    mode='nearest')
                # outputs_atlas_affine = F.interpolate(outputs_atlas_affine, size=(inputs_shape[2], inputs_shape[3], inputs_shape[4]),
                #                                      mode='trilinear', align_corners=True)
                outputs_atlas_mask_affine = F.interpolate(outputs_atlas_mask_affine,
                                                          size=(inputs_shape[2], inputs_shape[3], inputs_shape[4]),
                                                          mode='nearest')

                # outputs = outputs_atlas_mask
                #
                print("Spend Time:", time.time() - start_time)
                #
                # ## final mask
                # fullsize = outputs
                # #binarize output
                # wt = fullsize.chunk(target_class, dim=1)
                # wt = list(wt)
                # # wt = wt[0]
                # wt_num = len(wt)
                # s = fullsize.shape
                # for wn in range(wt_num):
                #     wt[wn] = (wt[wn] > 0.3).view(s[2], s[3], s[4])
                #
                # result = fullsize.new_zeros((s[2], s[3], s[4]), dtype=torch.uint8)
                # for wn in range(wt_num):
                #     result[wt[wn]] = wn+1
                #
                # npResult = result[:, :, :].cpu().numpy()
                # npResult = np.transpose(npResult, [1, 0, 2])
                # path = os.path.join(basePath, "{}_0_result.nii.gz".format(pid[0]))
                # utils.save_nii(path, npResult, None, None)

                ## affine mask
                fullsize = outputs_atlas_mask_affine
                # binarize output
                wt = fullsize.chunk(target_class, dim=1)
                wt = list(wt)
                # wt = wt[0]
                wt_num = len(wt)
                s = fullsize.shape
                for wn in range(wt_num):
                    wt[wn] = (wt[wn] > 0.3).view(s[2], s[3], s[4])

                result = fullsize.new_zeros((s[2], s[3], s[4]), dtype=torch.uint8)
                for wn in range(wt_num):
                    result[wt[wn]] = wn + 1

                npResult = result[:, :, :].cpu().numpy()
                npResult = np.transpose(npResult, [1, 0, 2])
                path = os.path.join(basePath, "{}_0_result_affine.nii.gz".format(pid[0]))
                utils.save_nii(path, npResult, None, None)

                # ## final image
                # result = outputs_atlas[0, 0, :, :, :]
                # npResult = result[:, :, :].cpu().numpy()
                # npResult = np.transpose(npResult, [1, 0, 2])
                # path = os.path.join(basePath, "{}_0_result_image.nii.gz".format(pid[0]))
                # utils.save_nii(path, npResult, None, None)
                # fullsize = outputs

                ## affine image
                # result = outputs_atlas_affine[0, 0, :, :, :]
                # npResult = result[:, :, :].cpu().numpy()
                # npResult = np.transpose(npResult, [1, 0, 2])
                # path = os.path.join(basePath, "{}_0_result_image_affine.nii.gz".format(pid[0]))
                # utils.save_nii(path, npResult, None, None)

        print("Done :)")

    def find_lr(self):
        expConfig = self.expConfig

        print('======= FINDING LEARNING RATE =======')
        print("ID: {}".format(expConfig.id))
        print('=====================================')

        epoch = 1
        while epoch < 10 and epoch <= self.bestMovingAvgEpoch + self.EARLY_STOPPING_AFTER_EPOCHS:

            expConfig.net = expConfig.NoNewReversible_affine()
            expConfig.net = expConfig.net.to(self.device)
            expConfig.INITIAL_LR = 1e-9 * 10 ** (epoch - 1)
            expConfig.optimizer = optim.AdamW(expConfig.net.parameters(), lr=expConfig.INITIAL_LR)
            expConfig.optimizer.zero_grad()

            running_loss = 0.0
            epoch_running_loss = 0.0
            startTime = time.time()

            # set net up training
            self.expConfig.net.train()
            self.mode = "train"
            torch.cuda.empty_cache()

            for i, data in enumerate(self.trainDataLoader):
                # load data
                inputs_list, pid, labels_list = data

                inputs, inputs_atlas, inputs_atlas_mask, labels = \
                    inputs_list[0], inputs_list[1], inputs_list[2], labels_list[0]

                inputs = F.interpolate(inputs, size=(224, 160, 96), mode='trilinear', align_corners=True)
                inputs_atlas = F.interpolate(inputs_atlas, size=(224, 160, 96), mode='trilinear', align_corners=True)
                inputs_atlas_mask = F.interpolate(inputs_atlas_mask, size=(224, 160, 96), mode='nearest')
                labels = F.interpolate(labels, size=(224, 160, 96), mode='nearest')

                inputs, inputs_atlas, inputs_atlas_mask, labels = \
                    inputs.to(self.device), inputs_atlas.to(self.device), inputs_atlas_mask.to(self.device), labels.to(
                        self.device)

                outputs_atlas_mask_affine = \
                    expConfig.net(inputs, inputs_atlas, inputs_atlas_mask)

                loss_dice, loss_mse = expConfig.loss([outputs_atlas_mask_affine],
                                                     [inputs, labels])

                print("Seg Loss:", loss_dice.item(), " MSE Loss:", loss_mse.item())#, " Dist Loss:", loss_dist.item())

                loss = loss_dice + loss_mse * 0.5# + loss_dist

                loss.backward()

                del inputs_list, labels_list, \
                    inputs, inputs_atlas, inputs_atlas_mask, labels, \
                    outputs_atlas_mask_affine, data

                # update params
                expConfig.optimizer.step()
                expConfig.optimizer.zero_grad()


                torch.cuda.empty_cache()

                total_loss = loss.item()

                del loss, loss_dice, loss_mse

                running_loss += total_loss
                epoch_running_loss += total_loss
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
            del expConfig.net, expConfig.optimizer
            torch.cuda.empty_cache()

    def train(self):
        expConfig = self.expConfig
        expConfig.net.to(self.device)
        print('============== TRAINING ==============')
        print("ID: {}".format(expConfig.id))
        print('======================================')

        epoch = self.startFromEpoch
        self.epoch_now = epoch

        resize_shape = [224, 160, 96]

        while epoch < expConfig.EPOCHS and epoch <= self.bestMovingAvgEpoch + self.EARLY_STOPPING_AFTER_EPOCHS:
            expConfig.net.train()
            self.mode = "train"

            expConfig.net = expConfig.net.to(self.device)
            expConfig.optimizer.zero_grad()

            running_loss = 0.0
            epoch_running_loss = 0.0
            startTime = time.time()

            for i, data in enumerate(self.trainDataLoader):

                # load data
                inputs_list, pid, labels_list = data

                inputs, inputs_atlas, inputs_atlas_mask, labels = \
                    inputs_list[0], inputs_list[1], inputs_list[2], labels_list[0]

                inputs_shape = inputs.shape

                resize_ratio = [0, 0, 0]

                resize_ratio[0] = inputs_shape[2] / resize_shape[0]
                resize_ratio[1] = inputs_shape[3] / resize_shape[1]
                resize_ratio[2] = inputs_shape[4] / resize_shape[2]

                final_resize_shape = [0, 0, 0]

                resize_arg = np.argmax(resize_ratio)
                for sss in range(3):
                    if resize_arg == sss:
                        final_resize_shape[resize_arg] = resize_shape[resize_arg]
                    else:
                        final_resize_shape[sss] = int(resize_shape[sss] / resize_ratio[sss])

                inputs = F.interpolate(inputs, size=final_resize_shape, mode='trilinear', align_corners=True)
                inputs_atlas = F.interpolate(inputs_atlas, size=final_resize_shape, mode='trilinear', align_corners=True)
                inputs_atlas_mask = F.interpolate(inputs_atlas_mask, size=final_resize_shape, mode='nearest')
                labels = F.interpolate(labels, size=final_resize_shape, mode='nearest')

                pad_z1 = int((resize_shape[2] - final_resize_shape[2]) / 2)
                pad_z2 = int((resize_shape[2] - final_resize_shape[2]) - pad_z1)
                pad_y1 = int((resize_shape[1] - final_resize_shape[1]) / 2)
                pad_y2 = int((resize_shape[1] - final_resize_shape[1]) - pad_y1)
                pad_x1 = int((resize_shape[0] - final_resize_shape[0]) / 2)
                pad_x2 = int((resize_shape[0] - final_resize_shape[0]) - pad_x1)

                inputs = F.pad(inputs, [pad_z1, pad_z2, pad_y1, pad_y2, pad_x1, pad_x2], "replicate")
                inputs_atlas = F.pad(inputs_atlas, [pad_z1, pad_z2, pad_y1, pad_y2, pad_x1, pad_x2], "replicate")
                inputs_atlas_mask = F.pad(inputs_atlas_mask, [pad_z1, pad_z2, pad_y1, pad_y2, pad_x1, pad_x2], "replicate")
                labels = F.pad(labels, [pad_z1, pad_z2, pad_y1, pad_y2, pad_x1, pad_x2], "constant")

                inputs, inputs_atlas, inputs_atlas_mask, labels = \
                    inputs.to(self.device), inputs_atlas.to(self.device), inputs_atlas_mask.to(self.device), labels.to(self.device)

                outputs_atlas, outputs_atlas_mask, outputs_atlas_affine, outputs_atlas_mask_affine = \
                    expConfig.net(inputs, inputs_atlas, inputs_atlas_mask)
                # outputs_atlas_mask_affine = \
                #     expConfig.net(inputs, inputs_atlas, inputs_atlas_mask)

                # loss_dice = expConfig.loss([outputs_atlas_mask_affine],
                #                            [inputs, labels])

                loss_dice, loss_mse, loss_dist = expConfig.loss([outputs_atlas,
                                                                 outputs_atlas_mask,
                                                                 outputs_atlas_affine,
                                                                 outputs_atlas_mask_affine],
                                                                [inputs, labels])

                print("Seg Loss:", loss_dice.item(), " MSE Loss:", loss_mse.item(), " Dist Loss:", loss_dist.item())

                loss = loss_dice + loss_mse * 0.5 + loss_dist * 1

                # torch.autograd.set_detect_anomaly(True)

                loss.backward()

                del inputs_list, labels_list, \
                    inputs, inputs_atlas, inputs_atlas_mask, labels, \
                    outputs_atlas_mask_affine, outputs_atlas, \
                    outputs_atlas_mask, outputs_atlas_affine

                for param_group in expConfig.optimizer.param_groups:
                    print("Current lr: {:.6f}".format(param_group['lr']))
                # update params
                expConfig.optimizer.step()
                expConfig.optimizer.zero_grad()

                print("GPU memory:", torch.cuda.max_memory_allocated(device=None))

                ## take lr sheudler step
                if hasattr(expConfig, "lr_sheudler"):
                    if isinstance(expConfig.lr_sheudler, optim.lr_scheduler.OneCycleLR):
                        expConfig.lr_sheudler.step()

                total_loss = loss.item()

                torch.cuda.empty_cache()

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

                if epoch % expConfig.VALIDATE_EVERY_K_EPOCHS == expConfig.VALIDATE_EVERY_K_EPOCHS - 1:
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
            self.epoch_now = epoch

        #print best mean dice
        print("ID:",expConfig.id)
        print("Best mean dice: {:.4f} at epoch {}".format(self.bestMeanDice, self.bestMeanDiceEpoch))

    def validate(self, epoch, target_class = 9):

        #set net up for inference
        expConfig = self.expConfig
        expConfig.net.eval()

        resize_shape = [224, 160, 96]

        self.mode = "eval"
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

                inputs_list, pid, labels_list = data

                inputs, inputs_atlas, inputs_atlas_mask, labels = \
                    inputs_list[0], inputs_list[1], inputs_list[2], labels_list[0]

                inputs_shape = inputs.shape

                resize_ratio = [0, 0, 0]

                resize_ratio[0] = inputs_shape[2] / resize_shape[0]
                resize_ratio[1] = inputs_shape[3] / resize_shape[1]
                resize_ratio[2] = inputs_shape[4] / resize_shape[2]

                final_resize_shape = [0, 0, 0]

                resize_arg = np.argmax(resize_ratio)
                for sss in range(3):
                    if resize_arg == sss:
                        final_resize_shape[resize_arg] = resize_shape[resize_arg]
                    else:
                        final_resize_shape[sss] = int(resize_shape[sss] / resize_ratio[sss])

                inputs = F.interpolate(inputs, size=final_resize_shape, mode='trilinear', align_corners=True)
                inputs_atlas = F.interpolate(inputs_atlas, size=final_resize_shape, mode='trilinear',
                                             align_corners=True)
                inputs_atlas_mask = F.interpolate(inputs_atlas_mask, size=final_resize_shape, mode='nearest')
                labels = F.interpolate(labels, size=final_resize_shape, mode='nearest')

                pad_z1 = int((resize_shape[2] - final_resize_shape[2]) / 2)
                pad_z2 = int((resize_shape[2] - final_resize_shape[2]) - pad_z1)
                pad_y1 = int((resize_shape[1] - final_resize_shape[1]) / 2)
                pad_y2 = int((resize_shape[1] - final_resize_shape[1]) - pad_y1)
                pad_x1 = int((resize_shape[0] - final_resize_shape[0]) / 2)
                pad_x2 = int((resize_shape[0] - final_resize_shape[0]) - pad_x1)

                inputs = F.pad(inputs, [pad_z1, pad_z2, pad_y1, pad_y2, pad_x1, pad_x2], "replicate")
                inputs_atlas = F.pad(inputs_atlas, [pad_z1, pad_z2, pad_y1, pad_y2, pad_x1, pad_x2], "replicate")
                inputs_atlas_mask = F.pad(inputs_atlas_mask, [pad_z1, pad_z2, pad_y1, pad_y2, pad_x1, pad_x2],
                                          "replicate")
                labels = F.pad(labels, [pad_z1, pad_z2, pad_y1, pad_y2, pad_x1, pad_x2], "constant")
                inputs, inputs_atlas, inputs_atlas_mask, labels = \
                    inputs.to(self.device), inputs_atlas.to(self.device), inputs_atlas_mask.to(self.device), labels.to(
                        self.device)

                # outputs_atlas_mask = \
                #     expConfig.net(inputs, inputs_atlas, inputs_atlas_mask)
                _, outputs_atlas_mask, _, _ = \
                    expConfig.net(inputs, inputs_atlas, inputs_atlas_mask)

                del inputs_list, labels_list, \
                    inputs, inputs_atlas, inputs_atlas_mask

                outputs = outputs_atlas_mask

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
        self.expConfig.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
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

    def getInputImage(self, inputs, labels, local_result, outputs_final_seg, mrf_i=0):
        ## Get input image from original image

        net_inputs_list = []
        net_labels_list = []
        net_mrf_list = []
        crop_size_list = []
        output_channels_list = []

        if mrf_i == 0:
            outputs_deform_new = copy.deepcopy(local_result)
            outputs_mrf = torch.zeros_like(local_result)
        else:
            if self.epoch_now < 50:
                outputs_deform_new = copy.deepcopy(local_result)
            else:
                outputs_deform_new = torch.from_numpy(outputs_final_seg.detach().cpu().numpy())
            outputs_mrf = torch.from_numpy(outputs_final_seg.detach().cpu().numpy())

        organs_size_flag = organs_size

        for oar_i in range(0, organs_num):

            organ_name_LR, organ_name = self._getOrganName(oar_i)
            organ_id_combine = organs_combine[organ_name_LR]

            ## get location map
            outputs_mrf_map = torch.zeros_like(outputs_mrf)
            labels_flag = 0
            for oic in range(len(organ_id_combine)):
                organ_id = organ_id_combine[oic]
                if oic == 0:
                    output_scores_map = outputs_deform_new[:, organ_id - 1:organ_id, :, :, :]
                    outputs_mrf_map[:, organ_id - 1:organ_id, :, :, :] = outputs_mrf[:, organ_id - 1:organ_id, :, :, :]

                else:
                    output_scores_map = output_scores_map + outputs_deform_new[:, organ_id - 1:organ_id, :, :, :]
                    outputs_mrf_map[:, organ_id - 1:organ_id, :, :, :] = outputs_mrf[:, organ_id - 1:organ_id, :, :, :]

                if torch.sum(labels[:, organ_id - 1:organ_id, :, :, :]).item() == 0:
                    labels_flag = 1

            if labels_flag:
                net_inputs_list.append(None)
                net_labels_list.append(None)
                net_mrf_list.append(None)
                output_channels_list.append(None)
                crop_size_list.append(None)
                continue

            crop_size = organs_size_flag[organ_name]
            output_channels_list.append(organs_channels[organ_name_LR])
            center = self._getCentroid(output_scores_map)
            mask_bbox = self._getCropImage(output_scores_map, center, crop_size)
            crop_size_list.append(mask_bbox)

            input_crop = torch.zeros([1, int(inputs.shape[1]), crop_size[3] + crop_size[0],
                                      crop_size[1] + crop_size[4], crop_size[2] + crop_size[5]])

            input_crop[:, :, :mask_bbox[5] - mask_bbox[2], :mask_bbox[4] - mask_bbox[1], :mask_bbox[3] - mask_bbox[0]] = \
                inputs[:, :, mask_bbox[2]:mask_bbox[5], mask_bbox[1]:mask_bbox[4], mask_bbox[0]:mask_bbox[3]]

            if labels is not None:
                label_crop = torch.zeros([1, int(labels.shape[1]), crop_size[3] + crop_size[0],
                                          crop_size[1] + crop_size[4], crop_size[2] + crop_size[5]])

                label_crop[:, :, :mask_bbox[5] - mask_bbox[2], :mask_bbox[4] - mask_bbox[1], :mask_bbox[3] - mask_bbox[0]] = \
                    labels[:, :, mask_bbox[2]:mask_bbox[5], mask_bbox[1]:mask_bbox[4], mask_bbox[0]:mask_bbox[3]]

            mrf_crop = torch.zeros([1, int(outputs_mrf_map.shape[1]), crop_size[3] + crop_size[0],
                                    crop_size[1] + crop_size[4], crop_size[2] + crop_size[5]])

            mrf_crop[:, :, :mask_bbox[5] - mask_bbox[2], :mask_bbox[4] - mask_bbox[1], :mask_bbox[3] - mask_bbox[0]] = \
                outputs_mrf_map[:, :, mask_bbox[2]:mask_bbox[5], mask_bbox[1]:mask_bbox[4], mask_bbox[0]:mask_bbox[3]]

            net_inputs_list.append(input_crop)
            if labels is not None:
                net_labels_list.append(label_crop)
            net_mrf_list.append(mrf_crop)
        return net_inputs_list, net_labels_list, net_mrf_list, crop_size_list, output_channels_list

    def _getOrganName(self, oar_i):
        ## get OAR name
        organ_name_LR = organs_combine_name[oar_i]
        if '-L' in organ_name_LR:
            organ_name = organ_name_LR.split('-L')[0]
        elif '-R' in organ_name_LR:
            organ_name = organ_name_LR.split('-R')[0]
        else:
            organ_name = organ_name_LR
        return organ_name_LR, organ_name

    def _getCentroid(self, matrix):
        matrix_sum = torch.sum(matrix)

        matrix_index = torch.ones([matrix.shape[2], matrix.shape[3], matrix.shape[4], 3])
        matrix_index = matrix_index

        matrix_index[:, :, :, 0] = torch.mul(matrix_index[:, :, :, 0],
                                             torch.reshape(torch.range(0, matrix.shape[2] - 1),
                                                           [matrix.shape[2], 1, 1]))
        matrix_index[:, :, :, 1] = torch.mul(matrix_index[:, :, :, 1],
                                             torch.reshape(torch.range(0, matrix.shape[3] - 1),
                                                           [1, matrix.shape[3], 1]))
        matrix_index[:, :, :, 2] = torch.mul(matrix_index[:, :, :, 2],
                                             torch.reshape(torch.range(0, matrix.shape[4] - 1),
                                                           [1, 1, matrix.shape[4]]))

        matrix_index_sum = torch.zeros([3])
        matrix_index_sum[0] = torch.sum(torch.mul(matrix_index[:, :, :, 0], matrix))
        matrix_index_sum[1] = torch.sum(torch.mul(matrix_index[:, :, :, 1], matrix))
        matrix_index_sum[2] = torch.sum(torch.mul(matrix_index[:, :, :, 2], matrix))

        center = (matrix_index_sum / matrix_sum)

        return center.int()

    def _getCropImage(self, matrix, center_point, crop_size):
        n_batch, channels, height, width, depth = matrix.shape

        centroid_z = center_point[2].item()
        centroid_y = center_point[1].item()
        centroid_x = center_point[0].item()

        if self.mode == "train":
            crop_size_i = crop_size[5] + crop_size[2]
            random_i = random.randint(int(-0.2 * crop_size_i), int(0.2 * crop_size_i))
            centroid_z = centroid_z + random_i
            crop_size_i = crop_size[4] + crop_size[1]
            random_i = random.randint(int(-0.2 * crop_size_i), int(0.2 * crop_size_i))
            centroid_y = centroid_y + random_i
            crop_size_i = crop_size[3] + crop_size[0]
            random_i = random.randint(int(-0.2 * crop_size_i), int(0.2 * crop_size_i))
            centroid_x = centroid_x + random_i

        z2 = np.minimum(depth, centroid_z + crop_size[5])# + 3)
        y2 = np.minimum(width, centroid_y + crop_size[4])# + 6)
        x2 = np.minimum(height, centroid_x + crop_size[3])# + 6)

        z1 = np.maximum(0, centroid_z - crop_size[2])# - 3)
        y1 = np.maximum(0, centroid_y - crop_size[1])# - 6)
        x1 = np.maximum(0, centroid_x - crop_size[0])# - 6)

        mask_bbox = [z1, y1, x1, z2, y2, x2]

        return mask_bbox