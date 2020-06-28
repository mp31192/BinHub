from comet_ml import Experiment, ExistingExperiment
import sys

sys.path.append("..")
import torch
import torch.optim as optim
import torch.nn as nn
import NPCUtils
import torch.nn.functional as F
import random
import numpy as np
import os
import time
from torch.nn import init
from torch.distributions.normal import Normal

# restore experiment
# VALIDATE_ALL = True
# PREDICT = True
# RESTORE_ID = '20200525090048_OARs_train_val_3D_1e-4_torch_deform_WL_aug_lr_dice_normal_dist_loss_affine_6temp_morechannels_2015_loadfrom51'
# RESTORE_EPOCH = 143
# LOG_COMETML_EXISTING_EXPERIMENT = ""

# general settings
SAVE_CHECKPOINTS = True  # set to true to create a checkpoint at every epoch
EXPERIMENT_TAGS = ["bugfreeFinalDrop"]
EXPERIMENT_NAME = "Reversible NO_NEW60, dropout"
EPOCHS = 300
BATCH_SIZE = 1
VIRTUAL_BATCHSIZE = 1
VALIDATE_EVERY_K_EPOCHS = 1
INPLACE = True

# hyperparameters
CHANNELS = [20, 30, 40, 50, 60]

# logging settings
LOG_EVERY_K_ITERATIONS = 1  # 0 to disable logging
LOG_MEMORY_EVERY_K_ITERATIONS = False
LOG_MEMORY_EVERY_EPOCH = True
LOG_EPOCH_TIME = True
LOG_VALIDATION_TIME = True
LOG_HAUSDORFF_EVERY_K_EPOCHS = 0  # must be a multiple of VALIDATE_EVERY_K_EPOCHS
LOG_COMETML = False
LOG_PARAMCOUNT = True
LOG_LR_EVERY_EPOCH = True

# data and augmentation
TRAIN_ORIGINAL_CLASSES = False  # train on original 5 classes
DATASET_WORKERS = 0
SOFT_AUGMENTATION = False  # Soft augmetation directly works on the 3 classes. Hard augmentation augments on the 5 orignal labels, then takes the argmax
NN_AUGMENTATION = True  # Has priority over soft/hard augmentation. Uses nearest-neighbor interpolation
DO_ROTATE = True
DO_SCALE = True
DO_FLIP = True
DO_ELASTIC_AUG = False
DO_INTENSITY_SHIFT = True

ROT_DEGREES = 20
SCALE_FACTOR = 1.1
SIGMA = 10
MAX_INTENSITY_SHIFT = 0.1

def loss(outputs, labels):
    return NPCUtils.AllLoss(outputs, labels)


def getChannelsAtIndex(index):
    if index < 0: index = 0
    if index >= len(CHANNELS): index = len(CHANNELS) - 1
    return CHANNELS[index]

class HeavisideStepFunction(nn.Module):
    def __init__(self, K=1500):
        super(HeavisideStepFunction, self).__init__()
        self.K = K

    def forward(self, input):
        output = 1 / (1 + torch.exp((-1 * input) * self.K))
        return output

class ResidualInner(nn.Module):
    def __init__(self, channels, kernel_size=3, groups_num=1, dilate_num=1):
        super(ResidualInner, self).__init__()
        # self.gn = nn.GroupNorm(groups, channels)
        # self.InN = nn.InstanceNorm3d(channels)
        self.SwN = SwitchNorm3d(channels, using_bn=False)
        real_k = kernel_size + (dilate_num - 1) * (kernel_size - 1)
        pad_num = np.int((real_k - 1) / 2)
        self.conv = nn.Conv3d(channels, channels, kernel_size, padding=pad_num, dilation=dilate_num, bias=False,
                              groups=groups_num)
        ##init.kaiming_normal(self.conv.weight)

    def forward(self, x):
        x = F.leaky_relu(self.SwN(self.conv(x)), negative_slope=0.2, inplace=INPLACE)
        return x


class ResidualInnerMultiview(nn.Module):
    def __init__(self, channels, kernel_size=(3, 3, 3), groups_num=1, dilate_num=1):
        super(ResidualInnerMultiview, self).__init__()
        # self.gn = nn.GroupNorm(groups, channels)
        # self.InN = nn.InstanceNorm3d(channels)
        self.SwN = SwitchNorm3d(channels, using_bn=False)
        self.dilate_num = (dilate_num, dilate_num, dilate_num)
        real_k = ()
        pad_num = ()
        for i in range(3):
            ks = kernel_size[i]
            dn = self.dilate_num[i]
            rk = ks + (dn - 1) * (ks - 1)
            real_k += (rk,)
            pad_num += (np.int((real_k[i] - 1) / 2),)
        self.conv = nn.Conv3d(channels, channels, kernel_size, padding=pad_num, dilation=self.dilate_num, bias=False,
                              groups=groups_num)
        init.kaiming_normal(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        x = F.leaky_relu(self.SwN(x), negative_slope=0.2, inplace=INPLACE)
        return x


class SEBlock(nn.Module):
    def __init__(self, inChannels):
        super(SEBlock, self).__init__()
        self.gap = nn.AdaptiveAvgPool3d([1, 1, 1])
        self.weightconv1 = nn.Conv3d(inChannels, inChannels // 4, 1)
        init.xavier_normal(self.weightconv1.weight)
        self.relu = nn.LeakyReLU(inplace=True)
        self.weightconv2 = nn.Conv3d(inChannels // 4, inChannels, 1)
        init.xavier_normal(self.weightconv2.weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h, w, z = x.size(2), x.size(3), x.size(4)
        x = self.gap(x)
        x = self.weightconv1(x)
        x = self.relu(x)
        x = self.weightconv2(x)
        # x = F.interpolate(input=x, size=(h, w, z), mode="trilinear", align_corners=False)
        x = self.sigmoid(x)
        return x


class ConvChangeDimension(nn.Module):
    def __init__(self, inChannels, outChannels, groups_num=1):
        super(ConvChangeDimension, self).__init__()
        self.convDimension = nn.Conv3d(inChannels, outChannels, 1, groups=groups_num)
        # init.kaiming_normal(self.convDimension.weight)
        # self.InN = nn.InstanceNorm3d(outChannels)
        self.SwN = SwitchNorm3d(outChannels, using_bn=False)

    def forward(self, x):
        x = F.leaky_relu(self.SwN(self.convDimension(x)), negative_slope=0.2, inplace=INPLACE)
        return x


class EncoderModule(nn.Module):
    def __init__(self, inChannels, outChannels, depth, downsample=True, kernel_size=(3, 3, 3),
                 res_flag=True, se_flag=True, groups_num=1, dilate_num=1):
        super(EncoderModule, self).__init__()
        self.downsample = downsample
        self.res_flag = res_flag
        self.se_flag = se_flag
        self.depth = depth
        self.pooling_size = ()
        self.coderBlocks = []
        for i in range(self.depth):
            self.coderBlocks.append(ResidualInnerMultiview(inChannels, kernel_size=kernel_size, groups_num=groups_num,
                                                           dilate_num=dilate_num))
        self.convBlocks = nn.ModuleList(self.coderBlocks)

        if self.se_flag:
            self.seblock = SEBlock(inChannels)

        self.convDimension = ConvChangeDimension(inChannels, outChannels, groups_num=groups_num)

        # pooling_size = ()
        for i in range(3):
            ps = kernel_size[i] - 1
            if ps == 0:
                ps = 1
            self.pooling_size += (ps,)

            # if downsample:
            #     self.pool3d = nn.Conv3d(outChannels,outChannels,self.pooling_size,stride=self.pooling_size, groups = groups_num)
            #     init.kaiming_normal(self.pool3d.weight)
            #     self.InNPool = nn.InstanceNorm3d(outChannels)

    def forward(self, x):
        if self.res_flag:
            x_res = x
        for i in range(self.depth):
            x = self.convBlocks[i](x)
        if self.se_flag:
            x = x * self.seblock(x)
        if self.res_flag:
            x = x + x_res
        x = self.convDimension(x)
        if self.downsample:
            # x = self.InNPool(self.pool3d(x))
            x = F.max_pool3d(x, self.pooling_size)
        return x


class DecoderModule(nn.Module):
    def __init__(self, inChannels, outChannels, depth, kernel_size=(3, 3, 3), upsample=True, res_flag=True,
                 groups_num=1):
        super(DecoderModule, self).__init__()
        self.res_flag = res_flag
        self.depth = depth
        self.upsample = upsample

        self.coderBlocks = []
        for i in range(self.depth):
            self.coderBlocks.append(ResidualInnerMultiview(inChannels, kernel_size=kernel_size, groups_num=groups_num))
        self.convBlocks = nn.ModuleList(self.coderBlocks)

        self.convDimension = ConvChangeDimension(inChannels, outChannels, groups_num=groups_num)

        self.pooling_size = ()
        for i in range(3):
            ps = kernel_size[i] - 1
            if ps == 0:
                ps = 1
            self.pooling_size += (ps,)

            # if self.upsample:
            #     self.deconv = nn.ConvTranspose3d(outChannels,outChannels,self.pooling_size,stride=self.pooling_size, groups = groups_num,
            #                                      bias=False)
            #     init.kaiming_normal_(self.deconv.weight)
            #     self.InNdeconv = nn.InstanceNorm3d(outChannels)

    def forward(self, x):
        if self.res_flag:
            x_res = x
        for i in range(self.depth):
            x = self.convBlocks[i](x)
        if self.res_flag:
            x = x + x_res
        x = self.convDimension(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=self.pooling_size, mode="trilinear", align_corners=False)
            # x = self.InNdeconv(self.deconv(x))
        return x

class EncoderMultiviewModule(nn.Module):
    def __init__(self, inChannels, outChannels, depth, downsample=True, kernel_size=(3, 3, 3), kernel_size_2 = (3, 3, 1),
                 res_flag=True, se_flag=True, groups_num=1, dilate_num=1):
        super(EncoderMultiviewModule, self).__init__()
        self.downsample = downsample
        self.res_flag = res_flag
        self.se_flag = se_flag
        self.depth = depth
        self.pooling_size = ()
        self.coderBlocks = []
        self.coderBlocks2 = []
        for i in range(self.depth):
            self.coderBlocks.append(ResidualInnerMultiview(inChannels, kernel_size=kernel_size, groups_num=groups_num,
                                                           dilate_num=dilate_num))
            self.coderBlocks2.append(ResidualInnerMultiview(inChannels, kernel_size=kernel_size_2, groups_num=groups_num,
                                                            dilate_num=dilate_num))
        self.convBlocks = nn.ModuleList(self.coderBlocks)
        self.convBlocks2 = nn.ModuleList(self.coderBlocks2)

        if self.se_flag:
            self.seblock = SEBlock(inChannels)

        self.convDimension = ConvChangeDimension(inChannels, outChannels, groups_num=groups_num)

        # pooling_size = ()
        for i in range(3):
            ps = kernel_size[i] - 1
            if ps == 0:
                ps = 1
            self.pooling_size += (ps,)

        # if downsample:
        #     self.pool3d = nn.Conv3d(outChannels,outChannels,self.pooling_size,stride=self.pooling_size, groups = groups_num)
        #     self.SwN = SwitchNorm3d(outChannels, using_bn=False)
            # init.kaiming_normal(self.pool3d.weight)
            # self.InNPool = nn.InstanceNorm3d(outChannels)

    def forward(self, x):
        if self.res_flag:
            x_res = x
        for i in range(self.depth):
            x = self.convBlocks[i](x) + self.convBlocks2[i](x)
        if self.se_flag:
            x = x * self.seblock(x)
        if self.res_flag:
            x = x + x_res
        x = self.convDimension(x)
        if self.downsample:
            # x = self.SwN(self.pool3d(x))
            x = F.max_pool3d(x, (2, 2, 2))#self.pooling_size)
        return x


class DecoderMultiviewModule(nn.Module):
    def __init__(self, inChannels, outChannels, depth, kernel_size=(3, 3, 3), kernel_size_2=(3, 3, 1), upsample=True, res_flag=True,
                 groups_num=1):
        super(DecoderMultiviewModule, self).__init__()
        self.res_flag = res_flag
        self.depth = depth
        self.upsample = upsample

        self.coderBlocks = []
        self.coderBlocks2 = []
        for i in range(self.depth):
            self.coderBlocks.append(ResidualInnerMultiview(inChannels, kernel_size=kernel_size, groups_num=groups_num))
            self.coderBlocks2.append(ResidualInnerMultiview(inChannels, kernel_size=kernel_size_2, groups_num=groups_num))
        self.convBlocks = nn.ModuleList(self.coderBlocks)
        self.convBlocks2 = nn.ModuleList(self.coderBlocks2)

        self.convDimension = ConvChangeDimension(inChannels, outChannels, groups_num=groups_num)

        self.pooling_size = ()
        for i in range(3):
            ps = kernel_size[i] - 1
            if ps == 0:
                ps = 1
            self.pooling_size += (ps,)

        # if self.upsample:
        #     self.deconv = nn.ConvTranspose3d(outChannels,outChannels,self.pooling_size,stride=self.pooling_size, groups = groups_num,
        #                                      bias=False)
        #     self.SwN = SwitchNorm3d(outChannels, using_bn=False)
            # init.kaiming_normal_(self.deconv.weight)
            # self.InNdeconv = nn.InstanceNorm3d(outChannels)

    def forward(self, x):
        if self.res_flag:
            x_res = x
        for i in range(self.depth):
            x = self.convBlocks[i](x) + self.coderBlocks2[i](x)
        if self.res_flag:
            x = x + x_res
        x = self.convDimension(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=(2, 2, 2), mode="trilinear", align_corners=False)
            # x = self.SwN(self.deconv(x))
        return x

class SwitchNorm3d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.997, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super(SwitchNorm3d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, D, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, D, H, W)
        return x * self.weight + self.bias

class CentroidLayer(nn.Module):
    def __init__(self):
        super(CentroidLayer, self).__init__()

    def forward(self, matrix):
        matrix_sum = torch.sum(matrix)

        matrix_index = torch.ones([matrix.shape[2], matrix.shape[3], matrix.shape[4], 3])
        matrix_index = matrix_index.cuda()

        matrix_index[:, :, :, 0] = torch.mul(matrix_index[:, :, :, 0],
                                             torch.reshape(torch.range(0, matrix.shape[2] - 1),
                                                           [matrix.shape[2], 1, 1]).cuda())
        matrix_index[:, :, :, 1] = torch.mul(matrix_index[:, :, :, 1],
                                             torch.reshape(torch.range(0, matrix.shape[3] - 1),
                                                           [1, matrix.shape[3], 1]).cuda())
        matrix_index[:, :, :, 2] = torch.mul(matrix_index[:, :, :, 2],
                                             torch.reshape(torch.range(0, matrix.shape[4] - 1),
                                                           [1, 1, matrix.shape[4]]).cuda())

        matrix_index_sum = torch.zeros([3]).cuda()
        matrix_index_sum[0] = torch.sum(torch.mul(matrix_index[:, :, :, 0], matrix))
        matrix_index_sum[1] = torch.sum(torch.mul(matrix_index[:, :, :, 1], matrix))
        matrix_index_sum[2] = torch.sum(torch.mul(matrix_index[:, :, :, 2], matrix))

        center = (matrix_index_sum / matrix_sum)

        return center.int()

class CropLayer(nn.Module):
    def __init__(self):
        super(CropLayer, self).__init__()

    def forward(self, matrix, center_point, crop_size):
        # center_point = center_point.cpu()
        n_batch, channels, height, width, depth = matrix.shape

        centroid_z = center_point[2].item()
        centroid_y = center_point[1].item()
        centroid_x = center_point[0].item()

        if self.training:
            # print(self.training)
            crop_size_i = crop_size[5] + crop_size[2]
            random_i = random.randint(int(-0.1 * crop_size_i), int(0.1 * crop_size_i))
            centroid_z = centroid_z + random_i
            crop_size_i = crop_size[4] + crop_size[1]
            random_i = random.randint(int(-0.1 * crop_size_i), int(0.1 * crop_size_i))
            centroid_y = centroid_y + random_i
            crop_size_i = crop_size[3] + crop_size[0]
            random_i = random.randint(int(-0.1 * crop_size_i), int(0.1 * crop_size_i))
            centroid_x = centroid_x + random_i

        # centroid_z = np.maximum(crop_size[2] + 6, centroid_z)
        # centroid_y = np.maximum(crop_size[1] + 6, centroid_y)
        # centroid_x = np.maximum(crop_size[0] + 6, centroid_x)
        #
        # centroid_z = np.minimum(depth - crop_size[5] - 6, centroid_z)
        # centroid_y = np.minimum(width - crop_size[4] - 6, centroid_y)
        # centroid_x = np.minimum(height - crop_size[3] - 6, centroid_x)
        #
        # mask_bbox = [centroid_z - crop_size[2] - 6,
        #              centroid_y - crop_size[1] - 6,
        #              centroid_x - crop_size[0] - 6,
        #              centroid_z + crop_size[5] + 6,
        #              centroid_y + crop_size[4] + 6,
        #              centroid_x + crop_size[3] + 6]

        z2 = np.minimum(depth, centroid_z + crop_size[5] + 3)
        y2 = np.minimum(width, centroid_y + crop_size[4] + 6)
        x2 = np.minimum(height, centroid_x + crop_size[3] + 6)

        z1 = np.maximum(0, centroid_z - crop_size[2] - 3)
        y1 = np.maximum(0, centroid_y - crop_size[1] - 6)
        x1 = np.maximum(0, centroid_x - crop_size[0] - 6)

        mask_bbox = [z1,
                     y1,
                     x1,
                     z2,
                     y2,
                     x2]

        return mask_bbox


class CropLayerAuto(nn.Module):
    def __init__(self):
        super(CropLayerAuto, self).__init__()

    def forward(self, matrix, center_point, crop_size):
        # center_point = center_point.cpu()
        n_batch, channels, height, width, depth = matrix.shape

        centroid_z = center_point[2]  # .item()
        centroid_y = center_point[1]  # .item()
        centroid_x = center_point[0]  # .item()

        centroid_z = torch.max(crop_size[2].int(), centroid_z)
        centroid_y = torch.max(crop_size[1].int(), centroid_y)
        centroid_x = torch.max(crop_size[0].int(), centroid_x)

        centroid_z = torch.min(depth - crop_size[5].int(), centroid_z)
        centroid_y = torch.min(width - crop_size[4].int(), centroid_y)
        centroid_x = torch.min(height - crop_size[3].int(), centroid_x)

        mask_bbox = [centroid_z - crop_size[2].int(),
                     centroid_y - crop_size[1].int(),
                     centroid_x - crop_size[0].int(),
                     centroid_z + crop_size[5].int(),
                     centroid_y + crop_size[4].int(),
                     centroid_x + crop_size[3].int()]

        return mask_bbox

class NoNewReversible_multiview_with_3d_mrf_nocrop(nn.Module):
    def __init__(self, Final_output=1, Input_Channels=1, kernel_size = (3, 3, 3), kernel_size_2 = (3, 3, 1)):
        super(NoNewReversible_multiview_with_3d_mrf_nocrop, self).__init__()
        depth = 2
        self.levels = 2
        self.Final_output = Final_output
        self.firstConvSeg = nn.Conv3d(Input_Channels + Final_output, CHANNELS[0], (1, 1, 1), padding=(0, 0, 0), bias=True)
        self.lastConv = nn.Conv3d(CHANNELS[0] * 2, Final_output, 1, bias=True)
        ## 3D segmentation
        # create encoder levels
        encoderModulesSeg = []
        for i in range(0, self.levels):
            if i == 0:
                downsample_flag = True
            else:
                downsample_flag = True
            encoderModulesSeg.append(
                EncoderMultiviewModule(CHANNELS[i], CHANNELS[i + 1], depth, downsample=downsample_flag,
                                       kernel_size=kernel_size, kernel_size_2=kernel_size_2,
                                       res_flag=True, se_flag=True, groups_num=1))
        self.encodersSeg = nn.ModuleList(encoderModulesSeg)

        self.MiddleConvSeg = nn.Conv3d(CHANNELS[i + 1], CHANNELS[i + 1], 1, bias=True)
        # self.MiddleInNSeg = nn.InstanceNorm3d(CHANNELS[i + 1])
        self.MiddleSwNSeg = SwitchNorm3d(CHANNELS[i + 1], using_bn=False)

        self.drop = nn.Dropout3d(p=0.2)

        # create decoder levels
        decoderModulesSeg = []
        for i in range(0, self.levels):
            if i == self.levels - 1:
                upsample_flag = True
            else:
                upsample_flag = True
            decoderModulesSeg.append(
                DecoderMultiviewModule(CHANNELS[self.levels - i] * 2,
                                       CHANNELS[self.levels - i - 1], depth,
                                       kernel_size=kernel_size, kernel_size_2=kernel_size_2,
                                       upsample=upsample_flag, res_flag=True, groups_num=1))
        self.decodersSeg = nn.ModuleList(decoderModulesSeg)

    def forward(self, input, mrf_result):
        ## Seperate image

        x_input = self.firstConvSeg(torch.cat([input, mrf_result], dim=1))
        x = x_input
        ## 3D segmentation
        inputStackSeg = []
        for i in range(self.levels):
            # inputStackSeg.append(x)
            x = self.encodersSeg[i](x)
            inputStackSeg.append(x)
            x = self.drop(x)

        x = F.leaky_relu(self.MiddleSwNSeg(self.MiddleConvSeg(x)))
        x = self.drop(x)

        for i in range(self.levels):
            x = torch.cat([x, inputStackSeg[self.levels - i - 1]], dim=1)
            x = self.decodersSeg[i](x)
            x = self.drop(x)

        x = torch.cat([x, x_input], dim=1)
        # x = self.OutConv1(x)
        output = self.lastConv(x)
        output = self.drop(output)
        output = F.sigmoid(output)
        return output

class NoNewReversible_multiview_with_3d_mrf_template_nocrop(nn.Module):
    def __init__(self, Final_output=1, Input_Channels=1, kernel_size = (3, 3, 3), kernel_size_2 = (3, 3, 1)):
        super(NoNewReversible_multiview_with_3d_mrf_template_nocrop, self).__init__()
        depth = 2
        self.levels = 2
        self.Final_output = Final_output
        self.firstConvSeg = nn.Conv3d(Input_Channels + Final_output, CHANNELS[0], (1, 1, 1), padding=(0, 0, 0), bias=True)
        self.firstConvSegTemp = nn.Conv3d(Final_output, CHANNELS[0], (1, 1, 1), padding=(0, 0, 0),
                                          bias=True)
        self.lastConv = nn.Conv3d(CHANNELS[0] * 2, Final_output, 1, bias=True)
        ## 3D segmentation
        # create encoder levels
        encoderModulesSeg = []
        for i in range(0, self.levels):
            if i == 0:
                downsample_flag = True
            else:
                downsample_flag = True
            encoderModulesSeg.append(
                EncoderMultiviewModule(CHANNELS[i], CHANNELS[i + 1], depth, downsample=downsample_flag,
                                       kernel_size=kernel_size, kernel_size_2=kernel_size_2,
                                       res_flag=True, se_flag=True, groups_num=1))
        self.encodersSeg = nn.ModuleList(encoderModulesSeg)

        self.MiddleConvSeg = nn.Conv3d(CHANNELS[i + 1], CHANNELS[i + 1], 1, bias=True)
        # self.MiddleInNSeg = nn.InstanceNorm3d(CHANNELS[i + 1])
        self.MiddleSwNSeg = SwitchNorm3d(CHANNELS[i + 1], using_bn=False)

        self.drop = nn.Dropout3d(p=0.2)

        # create decoder levels
        decoderModulesSeg = []
        for i in range(0, self.levels):
            if i == self.levels - 1:
                upsample_flag = True
            else:
                upsample_flag = True
            decoderModulesSeg.append(
                DecoderMultiviewModule(CHANNELS[self.levels - i] * 2,
                                       CHANNELS[self.levels - i - 1], depth,
                                       kernel_size=kernel_size, kernel_size_2=kernel_size_2,
                                       upsample=upsample_flag, res_flag=True, groups_num=1))
        self.decodersSeg = nn.ModuleList(decoderModulesSeg)

        ## 3D template segmentation
        # create encoder levels
        encoderModulesSegTemp = []
        for i in range(0, self.levels):
            if i == 0:
                downsample_flag = True
            else:
                downsample_flag = True
            encoderModulesSegTemp.append(
                EncoderMultiviewModule(CHANNELS[i], CHANNELS[i + 1], depth, downsample=downsample_flag,
                                       kernel_size=kernel_size, kernel_size_2=kernel_size_2,
                                       res_flag=True, se_flag=True, groups_num=10))
        self.encodersSegTemp = nn.ModuleList(encoderModulesSegTemp)

        self.MiddleConvSegTemp = nn.Conv3d(CHANNELS[i + 1], CHANNELS[i + 1], 1, bias=True)
        # self.MiddleInNSeg = nn.InstanceNorm3d(CHANNELS[i + 1])
        self.MiddleSwNSegTemp = SwitchNorm3d(CHANNELS[i + 1], using_bn=False)

        # create decoder levels
        decoderModulesSegTemp = []
        for i in range(0, self.levels):
            if i == self.levels - 1:
                upsample_flag = True
            else:
                upsample_flag = True
            decoderModulesSegTemp.append(
                DecoderMultiviewModule(CHANNELS[self.levels - i] * 2,
                                       CHANNELS[self.levels - i - 1], depth,
                                       kernel_size=kernel_size, kernel_size_2=kernel_size_2,
                                       upsample=upsample_flag, res_flag=True, groups_num=10))
        self.decodersSegTemp = nn.ModuleList(decoderModulesSegTemp)

    def forward(self, input, mrf_result, template):
        ## Seperate image

        x_input = self.firstConvSeg(torch.cat([input, mrf_result], dim=1))
        x = x_input

        x_temp = self.firstConvSegTemp(template)

        ## 3D segmentation
        inputStackSeg = []
        inputStackSegTemp = []
        for i in range(self.levels):
            # inputStackSeg.append(x)
            x = self.encodersSeg[i](x)
            x_temp = self.encodersSegTemp[i](x_temp)
            x = x + x_temp
            inputStackSeg.append(x)
            inputStackSegTemp.append(x_temp)
            x = self.drop(x)

        x = F.leaky_relu(self.MiddleSwNSeg(self.MiddleConvSeg(x)))
        x_temp = F.leaky_relu(self.MiddleConvSegTemp(self.MiddleConvSegTemp(x_temp)))
        x = x + x_temp
        x = self.drop(x)

        for i in range(self.levels):
            x = torch.cat([x, inputStackSeg[self.levels - i - 1]], dim=1)
            x = self.decodersSeg[i](x)
            x_temp = torch.cat([x_temp, inputStackSegTemp[self.levels - i - 1]], dim=1)
            x_temp = self.decodersSegTemp[i](x_temp)
            x = x + x_temp
            x = self.drop(x)

        x = torch.cat([x, x_input], dim=1)
        # x = self.OutConv1(x)
        output = self.lastConv(x)
        output = self.drop(output)
        output = F.sigmoid(output)
        return output

class NoNewReversible_multiview_with_3donly_mrf_nocrop(nn.Module):
    def __init__(self, Final_output=1, Input_Channels=1, kernel_size = (3, 3, 3), kernel_size_2 = (3, 3, 1)):
        super(NoNewReversible_multiview_with_3donly_mrf_nocrop, self).__init__()
        depth = 2
        self.levels = 2
        self.Final_output = Final_output
        self.firstConvSeg = nn.Conv3d(Input_Channels + 6, CHANNELS[0], (1, 1, 1), padding=(0, 0, 0), bias=True)
        self.lastConv = nn.Conv3d(CHANNELS[0] * 2, Final_output, 1, bias=True)
        ## 3D segmentation
        # create encoder levels
        encoderModulesSeg = []
        for i in range(0, self.levels):
            if i == 0:
                downsample_flag = True
            else:
                downsample_flag = True
            encoderModulesSeg.append(
                EncoderModule(CHANNELS[i], CHANNELS[i + 1], depth, downsample=downsample_flag,
                              kernel_size=kernel_size,
                              res_flag=True, se_flag=True, groups_num=1))
        self.encodersSeg = nn.ModuleList(encoderModulesSeg)

        self.MiddleConvSeg = nn.Conv3d(CHANNELS[i + 1], CHANNELS[i + 1], 1, bias=True)
        # self.MiddleInNSeg = nn.InstanceNorm3d(CHANNELS[i + 1])
        self.MiddleSwNSeg = SwitchNorm3d(CHANNELS[i + 1], using_bn=False)

        # create decoder levels
        decoderModulesSeg = []
        for i in range(0, self.levels):
            if i == self.levels - 1:
                upsample_flag = True
            else:
                upsample_flag = True
            decoderModulesSeg.append(
                DecoderModule(CHANNELS[self.levels - i] * 2,
                              CHANNELS[self.levels - i - 1], depth,
                              kernel_size=kernel_size,
                              upsample=upsample_flag, res_flag=True, groups_num=1))
        self.decodersSeg = nn.ModuleList(decoderModulesSeg)

    def forward(self, input, mrf_result):
        ## Seperate image

        x_input = self.firstConvSeg(torch.cat([input, mrf_result], dim=1))
        x = x_input
        ## 3D segmentation
        inputStackSeg = []
        for i in range(self.levels):
            # inputStackSeg.append(x)
            x = self.encodersSeg[i](x)
            # x = self.drop(x)
            inputStackSeg.append(x)

        x = F.leaky_relu(self.MiddleSwNSeg(self.MiddleConvSeg(x)))

        for i in range(self.levels):
            x = torch.cat([x, inputStackSeg[self.levels - i - 1]], dim=1)
            x = self.decodersSeg[i](x)
            # x = self.drop(x)

        x = torch.cat([x, x_input], dim=1)
        # x = self.OutConv1(x)
        output = self.lastConv(x)
        # output = self.drop(output)
        output = F.sigmoid(output)
        return output
