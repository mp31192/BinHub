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

# general settings
SAVE_CHECKPOINTS = True  # set to true to create a checkpoint at every epoch
EXPERIMENT_TAGS = ["bugfreeFinalDrop"]
EXPERIMENT_NAME = "Reversible NO_NEW60, dropout"
EPOCHS = 200
BATCH_SIZE = 1
VIRTUAL_BATCHSIZE = 1
VALIDATE_EVERY_K_EPOCHS = 1
INPLACE = True

# hyperparameters
CHANNELS = [16, 32, 64, 64, 64]

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
    return NPCUtils.AllLossRegSeg(outputs, labels)


def getChannelsAtIndex(index):
    if index < 0: index = 0
    if index >= len(CHANNELS): index = len(CHANNELS) - 1
    return CHANNELS[index]

class ResidualInner(nn.Module):
    def __init__(self, channels, kernel_size=3, groups_num=1, dilate_num=1):
        super(ResidualInner, self).__init__()
        # self.gn = nn.GroupNorm(groups, channels)
        self.InN = nn.InstanceNorm3d(channels)
        # self.SwN = SwitchNorm3d(channels, using_bn=False)
        real_k = kernel_size + (dilate_num - 1) * (kernel_size - 1)
        pad_num = np.int((real_k - 1) / 2)
        self.conv = nn.Conv3d(channels, channels, kernel_size, padding=pad_num, dilation=dilate_num, bias=False,
                              groups=groups_num)
        ##init.kaiming_normal(self.conv.weight)

    def forward(self, x):
        x = F.leaky_relu(self.InN(self.conv(x)), negative_slope=0.2, inplace=INPLACE)
        return x


class ResidualInnerMultiview(nn.Module):
    def __init__(self, channels, kernel_size=(3, 3, 3), groups_num=1, dilate_num=1):
        super(ResidualInnerMultiview, self).__init__()
        # self.gn = nn.GroupNorm(groups, channels)
        self.InN = nn.InstanceNorm3d(channels)
        # self.SwN = SwitchNorm3d(channels, using_bn=False)
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
        x = F.leaky_relu(self.InN(x), negative_slope=0.2, inplace=INPLACE)
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
        self.InN = nn.InstanceNorm3d(outChannels)
        # self.SwN = SwitchNorm3d(outChannels, using_bn=False)

    def forward(self, x):
        x = F.leaky_relu(self.InN(self.convDimension(x)), negative_slope=0.2, inplace=INPLACE)
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
        #     # init.kaiming_normal(self.pool3d.weight)
        #     self.SwNPool = SwitchNorm3d(outChannels, using_bn=False)

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
            # x = self.SwNPool(self.pool3d(x))
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
        #     # init.kaiming_normal_(self.deconv.weight)
        #     self.SwNdeconv = SwitchNorm3d(outChannels, using_bn=False)

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
            # x = self.SwNdeconv(self.deconv(x))
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
            x = F.max_pool3d(x, (2, 2, 1))#self.pooling_size)
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

        self.convDimension = ConvChangeDimension(inChannels, outChannels)#, groups_num=groups_num)

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
            x = F.interpolate(x, scale_factor=(2, 2, 1), mode="trilinear", align_corners=False)
            # x = self.SwN(self.deconv(x))
        return x

class SpatialTransformation(nn.Module):
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        super(SpatialTransformation, self).__init__()

    def meshgrid(self, height, width, depth):
        # x_t = torch.matmul(torch.ones([height, 1]), torch.transpose(torch.unsqueeze(torch.linspace(0.0, width - 1.0, width), 1), 1, 0))
        # y_t = torch.matmul(torch.unsqueeze(torch.linspace(0.0, height - 1.0, height), 1), torch.ones([1, width]))
        # y_t = torch.matmul(torch.unsqueeze(torch.linspace(0.0, height - 1.0, height), 1), torch.ones([1, width]))

        x_t = torch.ones([height, width, depth])
        x_t = torch.mul(x_t, torch.reshape(torch.range(0, height - 1), [height, 1, 1]))
        y_t = torch.ones([height, width, depth])
        y_t = torch.mul(y_t, torch.reshape(torch.range(0, width - 1), [1, width, 1]))
        z_t = torch.ones([height, width, depth])
        z_t = torch.mul(z_t, torch.reshape(torch.range(0, depth - 1), [1, 1, depth]))

        x_t = x_t.expand([height, width, depth])
        y_t = y_t.expand([height, width, depth])
        z_t = z_t.expand([height, width, depth])

        if self.use_gpu == True:
            x_t = x_t.cuda()
            y_t = y_t.cuda()
            z_t = z_t.cuda()

        return x_t, y_t, z_t

    def repeat(self, x, n_repeats):
        rep = torch.transpose(torch.unsqueeze(torch.ones(n_repeats), 1), 1, 0)
        rep = rep.long()
        x = torch.matmul(torch.reshape(x, (-1, 1)), rep)
        if self.use_gpu:
            x = x.cuda()

        return torch.squeeze(torch.reshape(x, (-1, 1)))

    def interpolate(self, im, x, y, z):

        im = F.pad(im, (0, 0, 1, 1, 1, 1, 1, 1, 0, 0))

        n_batch, height, width, depth, n_channel = im.shape

        n_batch, out_height, out_width, out_depth = x.shape

        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        z = z.reshape(1, -1)

        x = x + 1
        y = y + 1
        z = z + 1

        # print('x',torch.sum(x))
        # print('y',torch.sum(y))
        # print('z',torch.sum(z))

        max_x = height - 1
        max_y = width - 1
        max_z = depth - 1

        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1
        z0 = torch.floor(z).long()
        z1 = z0 + 1

        # print('flx',torch.sum(x))
        # print('fly',torch.sum(y))
        # print('flz',torch.sum(z))

        x0 = torch.clamp(x0, 0, max_x)
        x1 = torch.clamp(x1, 0, max_x)
        y0 = torch.clamp(y0, 0, max_y)
        y1 = torch.clamp(y1, 0, max_y)
        z0 = torch.clamp(z0, 0, max_z)
        z1 = torch.clamp(z1, 0, max_z)

        base = self.repeat(torch.arange(0, n_batch) * height * width * depth, out_height * out_width * out_depth)

        base_x0 = base + x0 * width * depth
        base_x1 = base + x1 * width * depth
        base00 = base_x0 + y0 * depth
        base01 = base_x0 + y1 * depth
        base10 = base_x1 + y0 * depth
        base11 = base_x1 + y1 * depth

        index000 = base00 + z0
        index001 = base00 + z1
        index010 = base01 + z0
        index011 = base01 + z1
        index100 = base10 + z0
        index101 = base10 + z1
        index110 = base11 + z0
        index111 = base11 + z1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = torch.reshape(im, [-1, n_channel])
        im_flat = im_flat.float()
        dim, _ = index000.transpose(1, 0).shape
        I000 = torch.gather(im_flat, 0, index000.transpose(1, 0).expand(dim, n_channel)).cpu()
        I001 = torch.gather(im_flat, 0, index001.transpose(1, 0).expand(dim, n_channel)).cpu()
        I010 = torch.gather(im_flat, 0, index010.transpose(1, 0).expand(dim, n_channel)).cpu()
        I011 = torch.gather(im_flat, 0, index011.transpose(1, 0).expand(dim, n_channel)).cpu()
        # print(torch.max(index100))
        # print(im_flat.shape)
        I100 = torch.gather(im_flat, 0, index100.transpose(1, 0).expand(dim, n_channel)).cpu()
        I101 = torch.gather(im_flat, 0, index101.transpose(1, 0).expand(dim, n_channel)).cpu()
        I110 = torch.gather(im_flat, 0, index110.transpose(1, 0).expand(dim, n_channel)).cpu()
        I111 = torch.gather(im_flat, 0, index111.transpose(1, 0).expand(dim, n_channel)).cpu()

        # print('x',torch.sum(x))
        # print('y',torch.sum(y))
        # print('z',torch.sum(z))

        x1_f = x1.float()
        y1_f = y1.float()
        z1_f = z1.float()

        # print('x1_f',torch.sum(x1_f))
        # print('y1_f',torch.sum(y1_f))
        # print('z1_f',torch.sum(z1_f))

        dx = x1_f - x
        dy = y1_f - y
        dz = z1_f - z

        # print('dx',torch.sum(dx))
        # print('dy',torch.sum(dy))
        # print('dz',torch.sum(dz))

        w111 = ((1.0 - dx) * (1.0 - dy) * (1.0 - dz))  # .permute(1,0)
        w110 = ((1.0 - dx) * (1.0 - dy) * dz)  # .permute(1,0)
        w101 = ((1.0 - dx) * dy * (1.0 - dz))  # .permute(1,0)
        w100 = ((1.0 - dx) * dy * dz)  # .permute(1,0)
        w011 = (dx * (1.0 - dy) * (1.0 - dz))  # .permute(1,0)
        w010 = (dx * (1.0 - dy) * dz)  # .permute(1,0)
        w001 = (dx * dy * (1.0 - dz))  # .permute(1,0)
        w000 = (dx * dy * dz)  # .permute(1,0)

        # w000 = (1.0-dx)*(1.0-dy)*(1.0-dz)#.permute(1,0)
        # w001 = ((1.0-dx)*(1.0-dy)*dz)#.permute(1,0)
        # w010 = ((1.0 - dx) * dy * (1.0 - dz))#.permute(1,0)
        # w011 = ((1.0 - dx) * dy * dz)#.permute(1,0)
        # w100 = (dx * (1.0 - dy) * (1.0 - dz))#.permute(1,0)
        # w101 = (dx * (1.0 - dy) * dz)#.permute(1,0)
        # w110 = (dx * dy * (1.0 - dz))#.permute(1,0)
        # w111 = (dx * dy * dz)#.permute(1,0)

        w111 = w111.permute(1, 0).cpu()
        w110 = w110.permute(1, 0).cpu()
        w101 = w101.permute(1, 0).cpu()
        w100 = w100.permute(1, 0).cpu()
        w011 = w011.permute(1, 0).cpu()
        w010 = w010.permute(1, 0).cpu()
        w001 = w001.permute(1, 0).cpu()
        w000 = w000.permute(1, 0).cpu()

        m000 = w000 * I000
        m001 = w001 * I001
        m010 = w010 * I010
        m011 = w011 * I011
        m100 = w100 * I100
        m101 = w101 * I101
        m110 = w110 * I110
        m111 = w111 * I111

        output = torch.sum(torch.squeeze(torch.stack([m000, m001, m010, m011,
                                                      m100, m101, m110, m111], dim=1)), 1)

        output = output.cuda()

        # output = torch.sum(torch.squeeze(torch.stack([w000 * I000, w001 * I001, w010 * I010, w011 * I011,
        #                                               w100 * I100, w101 * I101, w110 * I110, w111 * I111], dim = 1)), 1)
        output = torch.reshape(output, [n_batch, out_height, out_width, out_depth, n_channel])

        output = output.permute(0, 4, 1, 2, 3)

        return output

    def forward(self, moving_image, deformation_matrix):
        moving_image = moving_image.permute(0, 2, 3, 4, 1)
        deformation_matrix = deformation_matrix.permute(0, 2, 3, 4, 1)

        dx = deformation_matrix[:, :, :, :, 0]
        dy = deformation_matrix[:, :, :, :, 1]
        dz = deformation_matrix[:, :, :, :, 2]

        n_batch, height, width, depth = dx.shape

        x_mesh, y_mesh, z_mesh = self.meshgrid(height, width, depth)

        x_mesh = x_mesh.expand([n_batch, height, width, depth])
        y_mesh = y_mesh.expand([n_batch, height, width, depth])
        z_mesh = z_mesh.expand([n_batch, height, width, depth])

        x_new = dx + x_mesh
        y_new = dy + y_mesh
        z_new = dz + z_mesh

        return self.interpolate(moving_image, x_new, y_new, z_new)


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

class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """

    def __init__(self, size, mode='bilinear'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        # print(torch.max(grid[:, 0, :, :, :]).item())
        # print(torch.max(grid[:, 1, :, :, :]).item())
        # print(torch.max(grid[:, 2, :, :, :]).item())

        self.mode = mode

    def forward(self, src, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        new_locs = self.grid + flow

        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
            # print(torch.max(new_locs[:, i, ...]).item(), torch.min(new_locs[:, i, ...]).item())

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            # for i in range(len(shape)):
            #     print(torch.max(new_locs[:, :, :, :, i]).item(), torch.min(new_locs[:, :, :, :, i]).item())
            new_locs = new_locs[..., [2, 1, 0]]
            # for i in range(len(shape)):
            #     print(torch.max(new_locs[:, :, :, :, i]).item(), torch.min(new_locs[:, :, :, :, i]).item())

        return F.grid_sample(src, new_locs, mode=self.mode)

class SpatialTransformerAutoSize(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """

    def __init__(self, mode='bilinear'):
        """
        Instiatiate the block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformerAutoSize, self).__init__()

        # Create sampling grid
        # vectors = [torch.arange(0, s) for s in size]
        # grids = torch.meshgrid(vectors)
        # grid = torch.stack(grids)  # y, x, z
        # grid = torch.unsqueeze(grid, 0)  # add batch
        # grid = grid.type(torch.FloatTensor)
        # self.register_buffer('grid', grid)

        # print(torch.max(grid[:, 0, :, :, :]).item())
        # print(torch.max(grid[:, 1, :, :, :]).item())
        # print(torch.max(grid[:, 2, :, :, :]).item())

        self.mode = mode

    def forward(self, src, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """

        shape = flow.shape[2:]

        size = shape

        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # x, y, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor).cuda()
        # self.register_buffer('grid', grid)

        new_locs = grid + flow

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
            # print(torch.max(new_locs[:, i, ...]).item(), torch.min(new_locs[:, i, ...]).item())

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            # for i in range(len(shape)):
            #     print(torch.max(new_locs[:, :, :, :, i]).item(), torch.min(new_locs[:, :, :, :, i]).item())
            new_locs = new_locs[..., [2, 1, 0]]
            # for i in range(len(shape)):
            #     print(torch.max(new_locs[:, :, :, :, i]).item(), torch.min(new_locs[:, :, :, :, i]).item())

        return F.grid_sample(src, new_locs, mode=self.mode)

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

    def forward(self, matrix, center_point):
        # center_point = center_point.cpu()
        n_batch, channels, height, width, depth = matrix.shape

        centroid_z = center_point[2].item()
        centroid_y = center_point[1].item()
        centroid_x = center_point[0].item()

        # if self.training:
        #     # print(self.training)
        #     crop_size_i = crop_size[5] + crop_size[2]
        #     random_i = random.randint(int(-0.2 * crop_size_i), int(0.2 * crop_size_i))
        #     centroid_z = centroid_z + random_i
        #     crop_size_i = crop_size[4] + crop_size[1]
        #     random_i = random.randint(int(-0.2 * crop_size_i), int(0.2 * crop_size_i))
        #     centroid_y = centroid_y + random_i
        #     crop_size_i = crop_size[3] + crop_size[0]
        #     random_i = random.randint(int(-0.2 * crop_size_i), int(0.2 * crop_size_i))
        #     centroid_x = centroid_x + random_i

        bbox_index_list = torch.where(matrix[:, :, :, :, :] > 0.5)

        x1 = torch.min(bbox_index_list[2]).item()
        x2 = torch.max(bbox_index_list[2]).item()
        y1 = torch.min(bbox_index_list[3]).item()
        y2 = torch.max(bbox_index_list[3]).item()
        z1 = torch.min(bbox_index_list[4]).item()
        z2 = torch.max(bbox_index_list[4]).item()

        x1 = int((centroid_x - x1)) - 4
        x2 = int((x2 - centroid_x)) + 4
        y1 = int((centroid_y - y1)) - 4
        y2 = int((y2 - centroid_y)) + 4
        z1 = int((centroid_z - z1)) - 4
        z2 = int((z2 - centroid_z)) + 4

        c = 0
        while (x2 - x1) % 8:
            if c % 2:
                x2 += 1
            else:
                x1 -= 1
            c += 1

        c = 0
        while (y2 - y1) % 8:
            if c % 2:
                y2 += 1
            else:
                y1 -= 1
            c += 1

        c = 0
        while (z2 - z1) % 8:
            if c % 2:
                z2 += 1
            else:
                z1 -= 1
            c += 1


        final_x1 = np.maximum(int(centroid_x - x1), 0)
        final_y1 = np.maximum(int(centroid_y - y1), 0)
        final_z1 = np.maximum(int(centroid_z - z1), 0)
        final_x2 = np.minimum(int(centroid_x + x2), height)
        final_y2 = np.minimum(int(centroid_y + y2), width)
        final_z2 = np.minimum(int(centroid_z + z2), depth)

        mask_bbox = [final_x1, final_y1, final_z1, final_x2, final_y2, final_z2]

        return mask_bbox


class NoNewReversible_affine_class(nn.Module):
    def __init__(self):
        super(NoNewReversible_affine_class, self).__init__()
        depth = 2
        self.levels = 3
        self.patchsize = [224, 160, 96]

        self.firstConv = nn.Conv3d(4, CHANNELS[0], (1, 1, 1), padding=(0, 0, 0), bias=True)

        self.deform = SpatialTransformer(self.patchsize)

        ## 3D segmentation
        # create encoder levels
        encoderModulesSeg = []
        for i in range(0, self.levels):
            encoderModulesSeg.append(
                EncoderMultiviewModule(CHANNELS[i], CHANNELS[i + 1], depth, downsample=True,
                                       kernel_size=(3, 3, 3), kernel_size_2=(3, 3, 1),
                                       res_flag=True, se_flag=True, groups_num=1))
                # EncoderModule(CHANNELS[i], CHANNELS[i + 1], depth, downsample=True, kernel_size=(3, 3, 3),
                #               res_flag=True, se_flag=True, groups_num=1))
        self.encodersSeg = nn.ModuleList(encoderModulesSeg)

        self.MiddleConvSeg = nn.Conv3d(CHANNELS[i + 1], CHANNELS[i + 1], 1, bias=True)
        self.MiddleSwNSeg = nn.InstanceNorm3d(CHANNELS[i + 1])#SwitchNorm3d(CHANNELS[i + 1], using_bn=False)

        self.globalmaxpooling = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.FC_class = nn.Linear(CHANNELS[i + 1], 9)
        self.FC1 = nn.Linear(CHANNELS[i + 1], 16)
        self.FC2 = nn.Linear(16, 3 * 4)
        self.FC2.weight.data.zero_()
        self.FC2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, input, input_atlas, input_atlas_mask, input_atlas_bbox):

        _, _, x_shape, y_shape, z_shape = input.shape

        # input = F.interpolate(input, size=self.patchsize, mode='trilinear', align_corners=False)
        # input_atlas = F.interpolate(input_atlas, size=self.patchsize, mode='trilinear', align_corners=False)
        # input_atlas_mask = F.interpolate(input_atlas_mask, size=self.patchsize, mode='trilinear', align_corners=False)

        x = torch.cat([input, input_atlas, input_atlas_mask, input_atlas_bbox], dim=1)

        x = self.firstConv(x)

        inputStackSeg = []
        for i in range(self.levels):
            x = self.encodersSeg[i](x)
            inputStackSeg.append(x)

        x = F.leaky_relu(self.MiddleSwNSeg(self.MiddleConvSeg(x)))

        x_affine = self.globalmaxpooling(x)
        # x_class = x_affine.view(-1, 32)
        x_affine = x_affine.view(-1, 64)

        x_class = self.FC_class(x_affine)

        x_affine = self.FC1(x_affine)
        x_affine = self.FC2(x_affine)
        x_affine = x_affine.view(-1, 3, 4)

        affine_deform = F.affine_grid(x_affine, input.shape)

        outputs_atlas_affine = F.grid_sample(input_atlas, affine_deform.float(), mode='bilinear', padding_mode="border")
        outputs_atlas_box_affine = F.grid_sample(input_atlas_bbox, affine_deform.float(), mode='bilinear', padding_mode="border")
        outputs_atlas_mask_affine = F.grid_sample(input_atlas_mask, affine_deform.float(), mode='bilinear',
                                                 padding_mode="border")

        # outputs_atlas_affine = F.interpolate(outputs_atlas_affine, size=(x_shape, y_shape, z_shape),
        #                                      mode='trilinear', align_corners=False)
        # outputs_atlas_mask_affine = F.interpolate(outputs_atlas_mask_affine, size=(x_shape, y_shape, z_shape),
        #                                           mode='trilinear', align_corners=False)

        return outputs_atlas_affine, outputs_atlas_mask_affine, outputs_atlas_box_affine, x_class


class NoNewReversible_affine_class_seg(nn.Module):
    def __init__(self):
        super(NoNewReversible_affine_class_seg, self).__init__()
        depth = 2
        self.levels = 3
        self.patchsize = [224, 160, 96]

        self.firstConv = nn.Conv3d(4, CHANNELS[0], (1, 1, 1), padding=(0, 0, 0), bias=True)
        # self.decoderConv = nn.Conv3d(CHANNELS[0], CHANNELS[0], (3, 3, 3), padding=(1, 1, 1), bias=False)

        self.deform = SpatialTransformer(self.patchsize)

        ## 3D segmentation
        # create encoder levels
        encoderModulesSeg = []
        for i in range(0, self.levels):
            encoderModulesSeg.append(
                EncoderModule(CHANNELS[i], CHANNELS[i + 1], depth, downsample=True, kernel_size=(3, 3, 3),
                              res_flag=True, se_flag=True, groups_num=1))
        self.encodersSeg = nn.ModuleList(encoderModulesSeg)

        self.MiddleConvSeg = nn.Conv3d(CHANNELS[i + 1], CHANNELS[i + 1], 1, bias=True)
        # self.MiddleInNSeg = nn.InstanceNorm3d(CHANNELS_TEMP[i + 1])
        self.MiddleSwNSeg = nn.InstanceNorm3d(CHANNELS[i + 1])#SwitchNorm3d(CHANNELS[i + 1], using_bn=False)

        self.globalmaxpooling = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.FC_class = nn.Linear(CHANNELS[i + 1], 9)
        self.FC1 = nn.Linear(CHANNELS[i + 1], 16)
        self.FC2 = nn.Linear(16, 3 * 4)
        self.FC2.weight.data.zero_()
        self.FC2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))

        # for p in self.parameters():
        #     p.requires_grad = False

        self.deconv1 = nn.Conv3d(CHANNELS[i + 1], CHANNELS[i + 1] * 2, 1, bias=True)

        # create decoder levels
        decoderModulesSeg = []
        for i in range(0, self.levels):
            upsample_flag = True
            decoderModulesSeg.append(
                DecoderMultiviewModule(CHANNELS[self.levels - i] * 2 + CHANNELS[self.levels - i],
                                       CHANNELS[self.levels - i - 1] * 2, depth,
                                       kernel_size=(3, 3, 3), kernel_size_2=(3, 3, 1),
                                       upsample=upsample_flag, res_flag=True, groups_num=1))
        self.decodersSeg = nn.ModuleList(decoderModulesSeg)

        self.centroid = CentroidLayer()
        self.croplayer = CropLayer()

        self.lastConv = nn.Conv3d(CHANNELS[0] * 2, 9, 1, bias=True)

    def forward(self, input, input_atlas, input_atlas_mask, inputs_atlas_bbox):

        batchsize, _, x_shape, y_shape, z_shape = input.shape

        x = torch.cat([input, input_atlas, input_atlas_mask, inputs_atlas_bbox], dim=1)

        outputs_seg_final = torch.zeros((batchsize, 9, x_shape, y_shape, z_shape)).cuda()

        x = self.firstConv(x)

        inputStackSeg = []
        for i in range(self.levels):
            x = self.encodersSeg[i](x)
            inputStackSeg.append(x)

        x = F.leaky_relu(self.MiddleSwNSeg(self.MiddleConvSeg(x)))

        x_affine = self.globalmaxpooling(x)
        x_affine = x_affine.view(-1, 64)

        x_class = self.FC_class(x_affine)

        x_affine = self.FC1(x_affine)
        x_affine = self.FC2(x_affine)
        x_affine = x_affine.view(-1, 3, 4)

        affine_deform = F.affine_grid(x_affine, input.shape)

        outputs_atlas_affine = F.grid_sample(input_atlas, affine_deform, mode='bilinear', padding_mode="border")
        outputs_atlas_mask_affine = F.grid_sample(inputs_atlas_bbox, affine_deform, mode='bilinear', padding_mode="border")

        # outputs_atlas_affine = F.interpolate(outputs_atlas_affine, size=(x_shape, y_shape, z_shape),
        #                                      mode='trilinear', align_corners=False)
        # outputs_atlas_mask_affine = F.interpolate(outputs_atlas_mask_affine, size=(x_shape, y_shape, z_shape),
        #                                           mode='trilinear', align_corners=False)

        center_coord = self.centroid(outputs_atlas_mask_affine)
        crop_coord = self.croplayer(outputs_atlas_mask_affine, center_coord)

        x_crop = x[:, :, int(crop_coord[0] / 8):int(crop_coord[3] / 8),
                   int(crop_coord[1] / 8):int(crop_coord[4] / 8),
                   int(crop_coord[2] / 8):int(crop_coord[5] / 8)]

        x = self.deconv1(x_crop)

        for i in range(self.levels):
            x_input = inputStackSeg[self.levels - i - 1]
            down_num = 2 ** (self.levels - i)
            x_input = x_input[:, :, int(crop_coord[0] / down_num):int(crop_coord[3] / down_num),
                              int(crop_coord[1] / down_num):int(crop_coord[4] / down_num),
                              int(crop_coord[2] / down_num):int(crop_coord[5] / down_num)]
            x_input = F.interpolate(x_input,
                                        size=(x.shape[2],
                                              x.shape[3],
                                              x.shape[4]),
                                        mode='trilinear', align_corners=False)
            x = torch.cat([x, x_input], dim=1)
            x = self.decodersSeg[i](x)

        outputs_seg = self.lastConv(x)
        outputs_seg = F.sigmoid(outputs_seg)

        outputs_seg = F.interpolate(outputs_seg,
                                    size=(crop_coord[3] - crop_coord[0],
                                          crop_coord[4] - crop_coord[1],
                                          crop_coord[5] - crop_coord[2]),
                                    mode='trilinear', align_corners=False)

        outputs_seg_final[:, :, crop_coord[0]:crop_coord[3], crop_coord[1]:crop_coord[4], crop_coord[2]:crop_coord[5]] = \
            outputs_seg_final[:, :, crop_coord[0]:crop_coord[3], crop_coord[1]:crop_coord[4], crop_coord[2]:crop_coord[5]] + outputs_seg

        return outputs_atlas_affine, outputs_atlas_mask_affine, x_class, outputs_seg_final, crop_coord


class NoNewReversible_affine_class_seg_highresolution(nn.Module):
    def __init__(self):
        super(NoNewReversible_affine_class_seg_highresolution, self).__init__()
        depth = 2
        self.levels = 3
        self.patchsize = [224, 160, 96]

        self.firstConv = nn.Conv3d(4, CHANNELS[0], (1, 1, 1), padding=(0, 0, 0), bias=True)
        # self.decoderConv = nn.Conv3d(CHANNELS[0], CHANNELS[0], (3, 3, 3), padding=(1, 1, 1), bias=False)

        self.deform = SpatialTransformer(self.patchsize)

        ## 3D segmentation
        # create encoder levels
        encoderModulesSeg = []
        for i in range(0, self.levels):
            encoderModulesSeg.append(
                EncoderModule(CHANNELS[i], CHANNELS[i + 1], depth, downsample=True, kernel_size=(3, 3, 3),
                              res_flag=True, se_flag=True, groups_num=1))
        self.encodersSeg = nn.ModuleList(encoderModulesSeg)

        self.MiddleConvSeg = nn.Conv3d(CHANNELS[i + 1], CHANNELS[i + 1], 1, bias=True)
        self.MiddleSwNSeg = nn.InstanceNorm3d(CHANNELS[i + 1])#SwitchNorm3d(CHANNELS[i + 1], using_bn=False)

        self.globalmaxpooling = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.FC_class = nn.Linear(CHANNELS[i + 1], 9)
        self.FC1 = nn.Linear(CHANNELS[i + 1], 16)
        self.FC2 = nn.Linear(16, 3 * 4)
        self.FC2.weight.data.zero_()
        self.FC2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))

        # for p in self.parameters():
        #     p.requires_grad = False

        self.deconv1 = nn.Conv3d(1, CHANNELS[0] * 1, 1, bias=True)

        encoderModulesSeg2 = []
        for i in range(0, self.levels):
            downsample_flag = True
            encoderModulesSeg2.append(
                EncoderMultiviewModule(CHANNELS[i] * 2, CHANNELS[i + 1] * 1, depth, downsample=downsample_flag,
                                       kernel_size=(3, 3, 3), kernel_size_2=(3, 3, 1),
                                       res_flag=True, se_flag=True, groups_num=1))
        self.encodersSeg2 = nn.ModuleList(encoderModulesSeg2)

        self.MiddleConvSeg2 = nn.Conv3d(CHANNELS[i + 1] * 1, CHANNELS[i + 1] * 1, 1, bias=True)
        self.MiddleSwNSeg2 = nn.InstanceNorm3d(CHANNELS[i + 1])#SwitchNorm3d(CHANNELS[i + 1] * 1, using_bn=False)

        # create decoder levels
        decoderModulesSeg = []
        for i in range(0, self.levels):
            upsample_flag = True
            decoderModulesSeg.append(
                DecoderMultiviewModule(CHANNELS[self.levels - i] * 2,
                                       CHANNELS[self.levels - i - 1] * 1, depth,
                                       kernel_size=(3, 3, 3), kernel_size_2=(3, 3, 1),
                                       upsample=upsample_flag, res_flag=True, groups_num=1))
        self.decodersSeg = nn.ModuleList(decoderModulesSeg)

        self.centroid = CentroidLayer()
        self.croplayer = CropLayer()

        self.lastConv = nn.Conv3d(CHANNELS[0] * 1, 6, 1, bias=True)

    def forward(self, input, input_atlas, input_atlas_mask, inputs_atlas_bbox):

        batchsize, _, x_shape, y_shape, z_shape = input.shape

        x = torch.cat([input, input_atlas, input_atlas_mask, inputs_atlas_bbox], dim=1)

        outputs_seg_final = torch.zeros((batchsize, 6, x_shape, y_shape, z_shape)).cuda()

        x = self.firstConv(x)

        inputStackSeg = []
        inputStackSeg.append(x)
        for i in range(self.levels):
            x = self.encodersSeg[i](x)
            inputStackSeg.append(x)

        x = F.leaky_relu(self.MiddleSwNSeg(self.MiddleConvSeg(x)))

        x_affine = self.globalmaxpooling(x)
        x_affine = x_affine.view(-1, 64)

        x_class = self.FC_class(x_affine)

        x_affine = self.FC1(x_affine)
        x_affine = self.FC2(x_affine)
        x_affine = x_affine.view(-1, 3, 4)

        affine_deform = F.affine_grid(x_affine, input.shape)

        outputs_atlas_affine = F.grid_sample(input_atlas, affine_deform.float(), mode='bilinear', padding_mode="border")
        outputs_atlas_mask_affine = F.grid_sample(inputs_atlas_bbox, affine_deform.float(), mode='bilinear', padding_mode="border")

        # outputs_atlas_affine = F.interpolate(outputs_atlas_affine, size=(x_shape, y_shape, z_shape),
        #                                      mode='trilinear', align_corners=False)
        # outputs_atlas_mask_affine = F.interpolate(outputs_atlas_mask_affine, size=(x_shape, y_shape, z_shape),
        #                                           mode='trilinear', align_corners=False)

        center_coord = self.centroid(outputs_atlas_mask_affine)
        crop_coord = self.croplayer(outputs_atlas_mask_affine, center_coord)

        x_crop = input[:, :, crop_coord[0]:crop_coord[3], crop_coord[1]:crop_coord[4], crop_coord[2]:crop_coord[5]]
        # x_crop = F.interpolate(x_crop, size=(48, 48, 48),
        #                        mode='trilinear', align_corners=False)

        x = self.deconv1(x_crop)

        inputStackSeg2 = []
        for i in range(self.levels):
            x_input = inputStackSeg[i]
            down_num = 2 ** i
            x_input = x_input[:, :, int(crop_coord[0] / down_num):int(crop_coord[3] / down_num),
                              int(crop_coord[1] / down_num):int(crop_coord[4] / down_num),
                              int(crop_coord[2] / down_num):int(crop_coord[5] / down_num)]

            x_input = F.interpolate(x_input, size=(x.shape[2], x.shape[3], x.shape[4]),
                                    mode='trilinear', align_corners=False)

            x = torch.cat([x, x_input], dim=1)

            x = self.encodersSeg2[i](x)
            inputStackSeg2.append(x)

        x = F.leaky_relu(self.MiddleSwNSeg2(self.MiddleConvSeg2(x)))

        for i in range(self.levels):
            x_input2 = inputStackSeg2[self.levels - i - 1]
            x_input2 = F.interpolate(x_input2, size=(x.shape[2], x.shape[3], x.shape[4]),
                                    mode='trilinear', align_corners=False)
            x = torch.cat([x, x_input2], dim=1)
            x = self.decodersSeg[i](x)

        outputs_seg = self.lastConv(x)
        outputs_seg = F.sigmoid(outputs_seg)

        outputs_seg = F.interpolate(outputs_seg,
                                    size=(crop_coord[3] - crop_coord[0],
                                          crop_coord[4] - crop_coord[1],
                                          crop_coord[5] - crop_coord[2]),
                                    mode='trilinear', align_corners=False)

        # outputs_seg_final[:, :, crop_coord[0]:crop_coord[3], crop_coord[1]:crop_coord[4], crop_coord[2]:crop_coord[5]] = \
        #     outputs_seg_final[:, :, crop_coord[0]:crop_coord[3], crop_coord[1]:crop_coord[4], crop_coord[2]:crop_coord[5]] + outputs_seg

        outputs_seg_final[:, :, crop_coord[0]:crop_coord[3], crop_coord[1]:crop_coord[4],
                          crop_coord[2]:crop_coord[5]] = outputs_seg

        return outputs_atlas_affine, outputs_atlas_mask_affine, x_class, outputs_seg_final, crop_coord