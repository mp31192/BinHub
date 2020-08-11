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
from torch.cuda.amp.autocast_mode import autocast

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
        self.crop_pad_voxel = [0, 0, 0, 0, 0, 0]

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

        resize_shape = [224, 160, 96]

        with torch.no_grad():
            for i, data in enumerate(self.testDataLoader):
                start_time = time.time()

                inputs_list, pid = data

                inputs, inputs_atlas, inputs_atlas_mask = \
                    inputs_list[0], inputs_list[1], inputs_list[2]

                inputs_shape = inputs.shape
                outputs = torch.zeros([inputs_shape[0], 9, inputs_shape[2], inputs_shape[3], inputs_shape[4]])

                net_inputs_list, net_inputs_atlas_list, net_inputs_atlas_mask_list, \
                net_inputs_atlas_bbox_list, _, _ = \
                    self.getInputImage(inputs, inputs_atlas, inputs_atlas_mask, None)

                outputs_atlas_mask = torch.zeros(net_inputs_atlas_mask_list[0].shape[0], 9,
                                                 net_inputs_atlas_mask_list[0].shape[2], net_inputs_atlas_mask_list[0].shape[3],
                                                 net_inputs_atlas_mask_list[0].shape[4]).to(self.device)

                for sl in range(len(net_inputs_atlas_list)):

                    inputs, inputs_atlas, inputs_atlas_mask, inputs_atlas_bbox = \
                        net_inputs_list[sl].to(self.device), net_inputs_atlas_list[sl].to(self.device), \
                        net_inputs_atlas_mask_list[sl].to(self.device), net_inputs_atlas_bbox_list[sl].to(self.device)

                    # _, _, x_shape, y_shape, z_shape = inputs.shape

                    _, _, _, outputs_seg, _ = \
                        expConfig.net(inputs, inputs_atlas,
                                      inputs_atlas_mask,
                                      inputs_atlas_bbox)

                    if sl == 3 or sl == 4:
                        sl_r = 3
                    elif sl == 5 or sl == 6:
                        sl_r = 4
                    elif sl == 7 or sl == 8:
                        sl_r = 5
                    else:
                        sl_r = sl

                    outputs_atlas_mask[:, sl, :, :, :] = outputs_seg[:, sl_r, :, :, :]

                x1_1 = np.maximum(self.crop_pad_voxel[0], 0)
                x2_1 = np.minimum(inputs_shape[2] - self.crop_pad_voxel[3], inputs_shape[2])
                y1_1 = np.maximum(self.crop_pad_voxel[1], 0)
                y2_1 = np.minimum(inputs_shape[3] - self.crop_pad_voxel[4], inputs_shape[3])
                z1_1 = np.maximum(self.crop_pad_voxel[2], 0)
                z2_1 = np.minimum(inputs_shape[4] - self.crop_pad_voxel[5], inputs_shape[4])

                outputs_shape = outputs_atlas_mask.shape

                x1_2 = np.maximum(- self.crop_pad_voxel[0], 0)
                x2_2 = np.minimum(outputs_shape[2] + self.crop_pad_voxel[3], outputs_shape[2])
                y1_2 = np.maximum(- self.crop_pad_voxel[1], 0)
                y2_2 = np.minimum(outputs_shape[3] + self.crop_pad_voxel[4], outputs_shape[3])
                z1_2 = np.maximum(- self.crop_pad_voxel[2], 0)
                z2_2 = np.minimum(outputs_shape[4] + self.crop_pad_voxel[5], outputs_shape[4])

                outputs[:, :, x1_1:x2_1, y1_1:y2_1, z1_1:z2_1] = outputs_atlas_mask[:, :, x1_2:x2_2, y1_2:y2_2, z1_2:z2_2]

                #
                print("Spend Time:", time.time() - start_time)

                ## affine mask
                fullsize = outputs
                # binarize output
                wt = fullsize.chunk(target_class, dim=1)
                wt = list(wt)
                # wt = wt[0]
                wt_num = len(wt)
                s = fullsize.shape
                for wn in range(wt_num):
                    wt[wn] = (wt[wn] > 0.5).view(s[2], s[3], s[4])

                result = fullsize.new_zeros((s[2], s[3], s[4]), dtype=torch.uint8)
                for wn in range(wt_num):
                    result[wt[wn]] = wn + 1

                npResult = result[:, :, :].cpu().numpy()
                npResult = np.transpose(npResult, [1, 0, 2])
                path = os.path.join(basePath, "{}_0_result.nii.gz".format(pid[0]))
                utils.save_nii(path, npResult, None, None)

                # ## final image
                # result = outputs_atlas[0, 0, :, :, :]
                # npResult = result[:, :, :].cpu().numpy()
                # npResult = np.transpose(npResult, [1, 0, 2])
                # path = os.path.join(basePath, "{}_0_result_image.nii.gz".format(pid[0]))
                # utils.save_nii(path, npResult, None, None)
                #
                # affine multi channel image
                # result = outputs_atlas_mask[0, :, :, :, :]
                # npResult = result.cpu().numpy()
                # npResult = np.transpose(npResult, [2, 1, 3, 0])
                # path = os.path.join(basePath, "{}_0_affine_multichannel.nii.gz".format(pid[0]))
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

        class_loss_function = nn.CrossEntropyLoss()

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

                net_inputs_list, net_inputs_atlas_list, net_inputs_atlas_mask_list, \
                net_inputs_atlas_bbox_list, net_labels_list, net_labels_box_list = \
                    self.getInputImage(inputs, inputs_atlas, inputs_atlas_mask, labels)

                total_loss = 0
                counter = 0
                count_num = 0
                for sl in range(len(net_labels_list)):
                    if net_labels_list[sl] is None:
                        continue
                    count_num += 1

                for sl in range(len(net_labels_list)):

                    if net_labels_list[sl] is None:
                        continue

                    inputs, inputs_atlas, inputs_atlas_mask, inputs_atlas_bbox, labels, labels_box = \
                        net_inputs_list[sl].to(self.device), net_inputs_atlas_list[sl].to(self.device), \
                        net_inputs_atlas_mask_list[sl].to(self.device), net_inputs_atlas_bbox_list[sl].to(self.device), \
                        net_labels_list[sl].to(self.device), net_labels_box_list[sl].to(self.device)

                    _, _, x_shape, y_shape, z_shape = inputs.shape

                    with autocast():
                        outputs_atlas_affine, outputs_atlas_mask_affine, outputs_class, outputs_seg, crop_coord = \
                            expConfig.net(inputs, inputs_atlas,
                                          inputs_atlas_mask,
                                          inputs_atlas_bbox)


                        class_loss = class_loss_function(outputs_class, torch.ones((1), dtype=torch.long).to(self.device) * sl)

                        if sl == 3 or sl == 4:
                            sl_r = 3
                        elif sl == 5 or sl == 6:
                            sl_r = 4
                        elif sl == 7 or sl == 8:
                            sl_r = 5
                        else:
                            sl_r = sl

                        loss_dice, loss_mse, loss_dist, loss_seg = \
                            expConfig.loss([outputs_atlas_affine, outputs_atlas_mask_affine, outputs_seg[:, sl_r:sl_r+1, :, :, :]],
                                           [inputs, labels[:, sl:sl+1, :, :, :], labels_box])

                    # print("Seg Loss:", loss_dice.item(), " MSE Loss:", loss_mse.item(),
                    #       " Dist Loss:", loss_dist.item(), " Cls Loss:", class_loss.item(),
                    #       " True Seg Loss:", loss_seg.item())

                    loss_reg = loss_dice + loss_dist * 5 + loss_mse * 0.5 + class_loss * 0.25
                    print("Seg Loss:", loss_dice.item(), " Reg Loss:", loss_reg.item(), "\n",
                          "MSE Loss:", loss_mse.item(), " Dist Loss:", loss_dist.item(),
                          " Cls Loss:", class_loss.item(), " True Seg Loss:", loss_seg.item())

                    loss = loss_seg# + loss_reg * 0.5# + loss_shape * 0.5

                    # torch.autograd.set_detect_anomaly(True)

                    loss.backward()

                    total_loss = total_loss + loss.item()

                    counter += 1

                    for param_group in expConfig.optimizer.param_groups:
                        print("Current lr: {:.6f}".format(param_group['lr']))
                    # update params
                    expConfig.optimizer.step()
                    expConfig.optimizer.zero_grad()

                    torch.cuda.empty_cache()

                del inputs_list, labels_list, \
                    inputs, inputs_atlas, inputs_atlas_mask, labels, \
                    outputs_atlas_mask_affine, outputs_atlas_affine, \
                    outputs_class, outputs_seg, crop_coord, \
                    inputs_atlas_bbox

                print("GPU memory:", torch.cuda.max_memory_allocated(device=None))

                ## take lr sheudler step
                if hasattr(expConfig, "lr_sheudler"):
                    if isinstance(expConfig.lr_sheudler, optim.lr_scheduler.OneCycleLR):
                        expConfig.lr_sheudler.step()

                total_loss = total_loss / counter

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
                else:
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

                net_inputs_list, net_inputs_atlas_list, net_inputs_atlas_mask_list, \
                net_inputs_atlas_bbox_list, net_labels_list, net_labels_box_list = \
                    self.getInputImage(inputs, inputs_atlas, inputs_atlas_mask, labels)

                outputs_atlas_mask = torch.zeros(net_labels_list[0].shape[0], 9,
                                                 net_labels_list[0].shape[2], net_labels_list[0].shape[3],
                                                 net_labels_list[0].shape[4]).to(self.device)
                labels_mid = torch.zeros(net_labels_list[0].shape[0], 9,
                                         net_labels_list[0].shape[2], net_labels_list[0].shape[3],
                                         net_labels_list[0].shape[4]).to(self.device)

                for sl in range(len(net_labels_list)):

                    if net_labels_list[sl] is None:
                        continue

                    inputs, inputs_atlas, inputs_atlas_mask, inputs_atlas_bbox, labels, labels_box = \
                        net_inputs_list[sl].to(self.device), net_inputs_atlas_list[sl].to(self.device), \
                        net_inputs_atlas_mask_list[sl].to(self.device), net_inputs_atlas_bbox_list[sl].to(self.device), \
                        net_labels_list[sl].to(self.device), net_labels_box_list[sl].to(self.device)

                    # _, _, x_shape, y_shape, z_shape = inputs.shape

                    _, _, _, outputs_seg, _ = \
                        expConfig.net(inputs, inputs_atlas,
                                      inputs_atlas_mask,
                                      inputs_atlas_bbox)

                    # outputs_atlas_affine = F.interpolate(outputs_atlas_affine, size=(x_shape, y_shape, z_shape),
                    #                                      mode='trilinear', align_corners=False)
                    # outputs_atlas_mask_affine = F.interpolate(outputs_atlas_mask_affine,
                    #                                           size=(x_shape, y_shape, z_shape),
                    #                                           mode='trilinear', align_corners=False)
                    # outputs_seg = F.interpolate(outputs_seg,
                    #                             size=(inputs_shape[2], inputs_shape[3], inputs_shape[4]),
                    #                             mode='trilinear', align_corners=False)

                    if sl == 3 or sl == 4:
                        sl_r = 3
                    elif sl == 5 or sl == 6:
                        sl_r = 4
                    elif sl == 7 or sl == 8:
                        sl_r = 5
                    else:
                        sl_r = sl

                    outputs_atlas_mask[:, sl, :, :, :] = outputs_seg[:, sl_r, :, :, :]
                    labels_mid[:, sl, :, :, :] = labels[:, sl, :, :, :]

                labels = labels_mid

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

    def getInputImage(self, inputs, inputs_atlas, inputs_atlas_mask, labels):
        ## Get input image from original image

        net_inputs_list = []
        net_inputs_atlas_list = []
        net_inputs_atlas_mask_list = []
        net_inputs_atlas_bbox_list = []
        net_labels_list = []
        net_labels_box_list = []

        # resize_shape = (224, 160, 96)
        resize_shape = (256, 192, 96)

        image_shape = inputs.shape[2:]

        #################### pad to same size ######################START

        shape_cha = image_shape[0] - resize_shape[0]
        if shape_cha < 0:
            shape_cha = abs(shape_cha)
            t1 = int(shape_cha / 2)
            t2 = int(shape_cha - t1)
            inputs = F.pad(inputs, (0, 0, 0, 0, t1, t2))

            self.crop_pad_voxel[0] = -1 * t1
            self.crop_pad_voxel[3] = -1 * t2

            if labels is not None:
                labels = F.pad(labels, (0, 0, 0, 0, t1, t2))
        else:
            shape_cha = abs(shape_cha)
            t1 = 0
            t2 = image_shape[0] - shape_cha
            inputs = inputs[:, :, t1:t2, :, :]

            self.crop_pad_voxel[0] = t1
            self.crop_pad_voxel[3] = shape_cha - t1

            if labels is not None:
                labels = labels[:, :, t1:t2, :, :]

        shape_cha = image_shape[1] - resize_shape[1]
        if shape_cha < 0:
            shape_cha = abs(shape_cha)
            t1 = int(shape_cha / 2)
            t2 = int(shape_cha - t1)
            inputs = F.pad(inputs, (0, 0, t1, t2, 0, 0))

            self.crop_pad_voxel[1] = -1 * t1
            self.crop_pad_voxel[4] = -1 * t2

            if labels is not None:
                labels = F.pad(labels, (0, 0, t1, t2, 0, 0))
        else:
            shape_cha = abs(shape_cha)
            t1 = int(shape_cha / 2)
            t2 = int(image_shape[1] - (shape_cha - t1))
            inputs = inputs[:, :, :, t1:t2, :]

            self.crop_pad_voxel[1] = t1
            self.crop_pad_voxel[4] = shape_cha - t1

            if labels is not None:
                labels = labels[:, :, :, t1:t2, :]

        shape_cha = image_shape[2] - resize_shape[2]
        if shape_cha < 0:
            shape_cha = abs(shape_cha)
            t1 = int(shape_cha / 2)
            t2 = int(shape_cha - t1)
            inputs = F.pad(inputs, (t1, t2, 0, 0, 0, 0))

            self.crop_pad_voxel[2] = -1 * t1
            self.crop_pad_voxel[5] = -1 * t2

            if labels is not None:
                labels = F.pad(labels, (t1, t2, 0, 0, 0, 0))
        else:
            shape_cha = abs(shape_cha)
            t1 = int(shape_cha / 2)
            t2 = int(image_shape[2] - (shape_cha - t1))
            inputs = inputs[:, :, :, :, t1:t2]

            self.crop_pad_voxel[2] = t1
            self.crop_pad_voxel[5] = shape_cha - t1

            if labels is not None:
                labels = labels[:, :, :, :, t1:t2]

        image_shape = inputs_atlas.shape[2:]

        shape_cha = image_shape[0] - resize_shape[0]
        if shape_cha < 0:
            shape_cha = abs(shape_cha)
            t1 = int(shape_cha / 2)
            t2 = int(shape_cha - t1)
            inputs_atlas = F.pad(inputs_atlas, (0, 0, 0, 0, t1, t2))
            inputs_atlas_mask = F.pad(inputs_atlas_mask, (0, 0, 0, 0, t1, t2))
        else:
            shape_cha = abs(shape_cha)
            t1 = 0
            t2 = image_shape[0] - shape_cha
            inputs_atlas = inputs_atlas[:, :, t1:t2, :, :]
            inputs_atlas_mask = inputs_atlas_mask[:, :, t1:t2, :, :]

        shape_cha = image_shape[1] - resize_shape[1]
        if shape_cha < 0:
            shape_cha = abs(shape_cha)
            t1 = int(shape_cha / 2)
            t2 = int(shape_cha - t1)
            inputs_atlas = F.pad(inputs_atlas, (0, 0, t1, t2, 0, 0))
            inputs_atlas_mask = F.pad(inputs_atlas_mask, (0, 0, t1, t2, 0, 0))
        else:
            shape_cha = abs(shape_cha)
            t1 = int(shape_cha / 2)
            t2 = int(image_shape[1] - (shape_cha - t1))
            inputs_atlas = inputs_atlas[:, :, :, t1:t2, :]
            inputs_atlas_mask = inputs_atlas_mask[:, :, :, t1:t2, :]

        shape_cha = image_shape[2] - resize_shape[2]
        if shape_cha < 0:
            shape_cha = abs(shape_cha)
            t1 = int(shape_cha / 2)
            t2 = int(shape_cha - t1)
            inputs_atlas = F.pad(inputs_atlas, (t1, t2, 0, 0, 0, 0))
            inputs_atlas_mask = F.pad(inputs_atlas_mask, (t1, t2, 0, 0, 0, 0))
        else:
            shape_cha = abs(shape_cha)
            t1 = int(shape_cha / 2)
            t2 = int(image_shape[2] - (shape_cha - t1))
            inputs_atlas = inputs_atlas[:, :, :, :, t1:t2]
            inputs_atlas_mask = inputs_atlas_mask[:, :, :, :, t1:t2]
        #################### pad to same size ######################END


        # inputs = F.interpolate(inputs, size=resize_shape, mode='trilinear', align_corners=True)
        # inputs_atlas = F.interpolate(inputs_atlas, size=resize_shape, mode='trilinear',
        #                              align_corners=True)
        # # inputs_atlas_mask = F.interpolate(inputs_atlas_mask, size=resize_shape, mode='nearest')
        # inputs_atlas_mask = F.interpolate(inputs_atlas_mask, size=resize_shape, mode='trilinear')
        # if labels is not None:
        #     labels = F.interpolate(labels, size=resize_shape, mode='nearest')

        inputs_atlas_box = torch.zeros_like(inputs_atlas_mask)
        if labels is not None:
            labels_box = torch.zeros_like(labels)
        for c in range(inputs_atlas_mask.shape[1]):
            if labels is not None:
                if torch.sum(labels[:, c, :, :, :]).item() == 0:
                    net_inputs_list.append(None)
                    net_labels_list.append(None)
                    net_inputs_atlas_list.append(None)
                    net_inputs_atlas_mask_list.append(None)
                    net_inputs_atlas_bbox_list.append(None)
                    net_labels_box_list.append(None)
                    continue

            inputs_atlas_mask_binary = copy.deepcopy(inputs_atlas_mask)
            # inputs_atlas_mask_binary[inputs_atlas_mask_binary >= 0.3] = 1
            # inputs_atlas_mask_binary[inputs_atlas_mask_binary < 0.3] = 0
            ## get atlas mask bbox
            atlas_index_list = torch.where(inputs_atlas_mask_binary[:, c, :, :, :] == 1)
            atlas_x1 = torch.min(atlas_index_list[1]) - 4
            atlas_x2 = torch.max(atlas_index_list[1]) + 4
            atlas_y1 = torch.min(atlas_index_list[2]) - 4
            atlas_y2 = torch.max(atlas_index_list[2]) + 4
            atlas_z1 = torch.min(atlas_index_list[3]) - 4
            atlas_z2 = torch.max(atlas_index_list[3]) + 4
            inputs_atlas_box[:, c, atlas_x1:atlas_x2, atlas_y1:atlas_y2, atlas_z1:atlas_z2] = 1

            if labels is not None:
                ## get label mask bbox
                label_index_list = torch.where(labels[:, c, :, :, :] == 1)
                label_x1 = torch.min(label_index_list[1]) - 4
                label_x2 = torch.max(label_index_list[1]) + 4
                label_y1 = torch.min(label_index_list[2]) - 4
                label_y2 = torch.max(label_index_list[2]) + 4
                label_z1 = torch.min(label_index_list[3]) - 4
                label_z2 = torch.max(label_index_list[3]) + 4
                labels_box[:, c, label_x1:label_x2, label_y1:label_y2, label_z1:label_z2] = 1

            ## add image to list
            net_inputs_list.append(inputs)
            net_inputs_atlas_list.append(inputs_atlas)
            net_inputs_atlas_mask_list.append(inputs_atlas_mask[:, c:c+1, :, :, :])
            net_inputs_atlas_bbox_list.append(inputs_atlas_box[:, c:c+1, :, :, :])
            if labels is not None:
                net_labels_list.append(labels)
                net_labels_box_list.append(labels_box[:, c:c + 1, :, :, :])

        return net_inputs_list, net_inputs_atlas_list, net_inputs_atlas_mask_list, \
               net_inputs_atlas_bbox_list, net_labels_list, net_labels_box_list