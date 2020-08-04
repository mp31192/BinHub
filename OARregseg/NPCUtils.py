import medpy.metric.binary as medpyMetrics
import numpy as np
import math
import torch
import torch.nn.functional as F

def softDice(pred, target, smoothing=1, nonSquared=False):
    intersection = (pred * target).sum(dim=(1, 2, 3))
    if nonSquared:
        union = (pred).sum(dim=(1, 2, 3)) + (target).sum(dim=(1, 2, 3))
    else:
        union = (pred * pred).sum(dim=(1, 2, 3)) + (target * target).sum(dim=(1, 2, 3))
    dice = (2 * intersection + smoothing) / (union + smoothing)

    #fix nans
    dice[dice != dice] = dice.new_tensor([1.0])

    return dice.mean()

def softDice2D(pred, target, smoothing=1, nonSquared=False):
    intersection = (pred * target).sum(dim=(1, 2))
    if nonSquared:
        union = (pred).sum() + (target).sum()
    else:
        union = (pred * pred).sum(dim=(1, 2)) + (target * target).sum(dim=(1, 2))
    dice = (2 * intersection + smoothing) / (union + smoothing)

    #fix nans
    dice[dice != dice] = dice.new_tensor([1.0])

    return dice.mean()

def LocationCentroid(matrix):
    if matrix.is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    matrix = matrix#.cpu()

    matrix_sum = torch.sum(matrix)

    matrix_index = torch.ones([matrix.shape[1], matrix.shape[2], matrix.shape[3], 3])
    matrix_index = matrix_index.to(device)

    matrix_index[:,:,:,0] = torch.mul(matrix_index[:,:,:,0], torch.reshape(torch.range(0,matrix.shape[1]-1), [matrix.shape[1], 1, 1]).to(device))
    matrix_index[:,:,:,1] = torch.mul(matrix_index[:,:,:,1], torch.reshape(torch.range(0,matrix.shape[2]-1), [1, matrix.shape[2], 1]).to(device))
    matrix_index[:,:,:,2] = torch.mul(matrix_index[:,:,:,2], torch.reshape(torch.range(0,matrix.shape[3]-1), [1, 1, matrix.shape[3]]).to(device))


    matrix_index_sum = torch.zeros([3]).to(device)
    matrix_index_sum[0] = torch.sum(torch.mul(matrix_index[:, :, :, 0], matrix))
    matrix_index_sum[1] = torch.sum(torch.mul(matrix_index[:, :, :, 1], matrix))
    matrix_index_sum[2] = torch.sum(torch.mul(matrix_index[:, :, :, 2], matrix))

    center = matrix_index_sum / (matrix_sum + 1)

    return center

def LocationLoss(pred, target):

    b, x_img, y_img, z_img = pred.shape

    predCentroid = LocationCentroid(pred)
    targetCentroid = LocationCentroid(target)

    normal_xyz = torch.Tensor([x_img, y_img, z_img]).cuda()

    # print("Normalized XYZ:", normal_xyz)
    # print("Pred Center:", predCentroid)
    # print("Target Center:", targetCentroid)

    predCentroidNormal = predCentroid.cuda() / normal_xyz
    targetCentroidNormal = targetCentroid.cuda() / normal_xyz

    Dis = targetCentroidNormal - predCentroidNormal
    Dis = torch.abs(Dis).float()
    return torch.sum(Dis)

def generalizeDice(pred,target,smoothing=1, nonSquared=False):
    intersection = (pred * target).sum(dim=(1,2,3))
    union = (pred*pred).sum(dim=(1,2,3)) + (target*target).sum(dim=(1,2,3))
    wl = 1/(target.sum(dim=(1,2,3))**2+1e-5)
    dice = (2 * intersection * wl + smoothing) / (wl * union + smoothing)
    # fix nans
    dice[dice != dice] = dice.new_tensor([1.0])
    return dice

def tverskyDice(pred,target,smoothing=1,nonSquared=False):
    alpha = 0.5
    beta = 0.5
    p1 = 1-pred
    g1 = 1-target
    num = (pred * target).sum(dim=(1,2,3))
    den = num + alpha * (pred * g1).sum(dim=(1,2,3)) + beta * (p1 * target).sum(dim=(1,2,3))
    T = num/(den+1e-6)
    return T

def tverskyDice2D(pred,target,smoothing=1,nonSquared=False):
    alpha = 0.5
    beta = 0.5
    p1 = 1-pred
    g1 = 1-target
    num = (pred * target).sum(dim=(1,2))
    den = num + alpha * (pred * g1).sum(dim=(1,2)) + beta * (p1 * target).sum(dim=(1,2))
    T = num/(den+1e-6)
    return T

def focalDice(pred,target,smoothing=1,nonSquared=False):
    alpha = 0.5
    beta = 0.5
    p1 = 1 - pred
    g1 = 1 - target
    num = (pred * target * (p1**2)).sum(dim=(1, 2, 3))
    den = num + alpha * (pred * g1).sum(dim=(1, 2, 3)) + beta * (p1 * target).sum(dim=(1, 2, 3))
    T = num / (den + 1e-6)
    return T



def crossentropy(pred,target):
    ce = F.binary_cross_entropy(pred, target)
    return ce

def focal(pred,target,smoothing=1,nonSquared=False):
    ancientone = torch.ones_like(pred)
    focal_re = - (torch.log(torch.clamp(pred,1e-6,1))*target*((1-pred)**2)).sum(dim=(1,2,3))/ancientone.sum(dim=(1,2,3))
    return focal_re

def dice(pred, target):
    predBin = (pred >= 0.5).float()
    return softDice(predBin, target, 0, True).item()

def dice2D(pred, target):
    predBin = (pred > 0.5).float()
    return softDice2D(predBin, target, 0, True).item()

def diceLoss(pred, target):
    return 1 - softDice(pred, target, nonSquared=True)

def dice2DLoss(pred, target, nonSquared=False):
    b, tx, ty, tz = target.shape
    dice2Dlist = []
    for zi in range(tz):
        if torch.sum(target[:, :, :, zi]).item() > 0:
            dice2D_i = softDice2D(pred[:,:,:,zi], target[:,:,:,zi], nonSquared=True)
            dice2Dlist.append(dice2D_i)
    return 1-(np.sum(dice2Dlist)/len(dice2Dlist))

def diceLossSDM(pred, target, nonSquared=False):
    return softDice(pred, target, nonSquared=True)

def productLoss(pred, target, smoothing = 1):
    ytpt = (pred * target).sum(dim=(1, 2, 3))
    pt2 = (pred * pred).sum(dim=(1, 2, 3))
    yt2 = (target * target).sum(dim=(1, 2, 3))
    loss = ytpt / (ytpt + pt2 + yt2 + smoothing)
    return loss

def L1Loss(pred, target):
    loss = torch.abs(pred - target)
    loss = loss.sum(dim=(1, 2, 3)) / (target.shape[1] * target.shape[2] * target.shape[3])
    return loss

def generdiceLoss(pred, target, nonSquared=False):
    return 1 - generalizeDice(pred, target, nonSquared=nonSquared)

def tverskydiceLoss(pred, target, nonSquared=False):
    return 1 - tverskyDice(pred, target, nonSquared=nonSquared)

def tverskydice2DLoss(pred, target, nonSquared=False):
    return 1 - tverskyDice2D(pred, target, nonSquared=nonSquared)

def focaldiceLoss(pred, target, nonSquared=False):
    return 1 - focalDice(pred, target, nonSquared=nonSquared)

def focaladddiceLoss(pred, target, nonSquared=False):
    # print(focal(pred,target))
    # print(1 - tverskyDice(pred, target, nonSquared=nonSquared))
    return (1 - focal(pred, target, nonSquared=nonSquared)) + (1 - tverskyDice(pred, target, nonSquared=nonSquared))

def ceadddiceLoss(pred, target, nonSquared=False):
    return crossentropy(pred, target) + (1 - tverskyDice(pred, target, nonSquared=nonSquared))


def Active_Contour_Loss(pred, target):
    """
    lenth term
    """

    # pred = pred.cpu()
    # target = target.cpu()
    target_sum = torch.sum(target)

    x = pred[:, 1:, :, :] - pred[:, :-1, :, :]
    y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    z = pred[:, :, :, 1:] - pred[:, :, :, :-1]

    delta_x = x[:, 1:, :-2, :-2] ** 2
    delta_y = y[:, :-2, 1:, :-2] ** 2
    delta_z = z[:, :-2, :-2, 1:] ** 2
    delta_u = torch.abs(delta_x + delta_y + delta_z)

    epsilon = 0.00000001  # where is a parameter to avoid square root is zero in practice.
    lenth_pred = torch.sum(torch.sqrt(delta_u + epsilon))  # equ.(11) in the paper

    x = target[:, 1:, :, :] - target[:, :-1, :, :]
    y = target[:, :, 1:, :] - target[:, :, :-1, :]
    z = target[:, :, :, 1:] - target[:, :, :, :-1]

    delta_x = x[:, 1:, :-2, :-2] ** 2
    delta_y = y[:, :-2, 1:, :-2] ** 2
    delta_z = z[:, :-2, :-2, 1:] ** 2
    delta_u = torch.abs(delta_x + delta_y + delta_z)

    epsilon = 0.00000001  # where is a parameter to avoid square root is zero in practice.
    lenth_target = torch.sum(torch.sqrt(delta_u + epsilon))  # equ.(11) in the paper

    lenth = torch.abs(lenth_pred - lenth_target)/lenth_pred

    """
    region term
    """

    C_1 = torch.ones_like(pred)
    C_2 = torch.zeros_like(pred)

    region_in = torch.abs(torch.sum(pred[:, :, :, :] * ((target[:, :, :, :] - C_1) ** 2))) / torch.sum(pred)  # equ.(12) in the paper
    region_out = torch.abs(torch.sum((1 - pred[:, :, :, :]) * ((target[:, :, :, :] - C_2) ** 2))) / torch.sum(1 - pred) # equ.(12) in the paper
    region = region_in + region_out

    print("length loss:", lenth)
    print("region loss:", region)

    lambdaP = 1  # lambda parameter could be various.
    w = 1

    loss = w * lenth + lambdaP * region

    # loss = loss.cuda()

    return loss

def Region_Loss(pred, target):

    """
    region term
    """

    C_1 = torch.ones_like(pred)
    C_2 = torch.zeros_like(pred)

    region_in = torch.abs(torch.sum(pred[:, :, :, :] * ((target[:, :, :, :] - C_1) ** 2))) / torch.sum(pred)  # equ.(12) in the paper
    region_out = torch.abs(torch.sum((1 - pred[:, :, :, :]) * ((target[:, :, :, :] - C_2) ** 2))) / torch.sum(1 - pred) # equ.(12) in the paper
    region = region_in + region_out

    # print("region loss:", region)

    loss = region

    # loss = loss.cuda()

    return loss

def DiceSDMLoss(outputs, labels, outputs_dist, dist):
    target_class = labels.shape[1]

    ## Seg
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

    ## SDM
    wt_dist = outputs_dist.chunk(target_class, dim=1)
    wt_dist = list(wt_dist)
    wt_num = len(wt_dist)
    s = wt_dist[0].shape
    for wn in range(wt_num):
        wt_dist[wn] = wt_dist[wn].view(s[0], s[2], s[3], s[4])

    wtMask_dist = dist.chunk(target_class, dim=1)
    wtMask_dist = list(wtMask_dist)
    s = wtMask_dist[0].shape
    for wn in range(wt_num):
        wtMask_dist[wn] = wtMask_dist[wn].view(s[0], s[2], s[3], s[4])


    ## Seg loss + SDM loss
    wtLoss_Dice = []
    wtLoss_product = []
    wtLoss_L1 = []
    for wn in range(wt_num):
        wtLoss_Dice_1 = diceLossSDM(wt[wn], wtMask[wn])  # + crossentropy(wt[wn], wtMask[wn])
        wtLoss_product_1 = productLoss(wt_dist[wn], wtMask_dist[wn])  # + crossentropy(wt[wn], wtMask[wn])
        wtLoss_L1_1 = L1Loss(wt_dist[wn], wtMask_dist[wn])  # + crossentropy(wt[wn], wtMask[wn])
        wtLoss_L1.append(wtLoss_L1_1)
        wtLoss_product.append(wtLoss_product_1)
        wtLoss_Dice.append(wtLoss_Dice_1)

    Lseg = target_class - np.sum(wtLoss_Dice)
    Lproduct = -1 * np.sum(wtLoss_product)
    L1 = np.sum(wtLoss_L1)
    print('Lseg:',Lseg,' Lpro:',Lproduct, ' L1:',L1)

    Loss = Lseg + 10 * (L1 + Lproduct)

    return Loss


def NPCDiceACLoss(outputs, labels, nonSquared=False, target_class = 22):

    target_class = labels.shape[1]

    #bring outputs into correct shape
    wt = outputs.chunk(target_class, dim=1)
    wt = list(wt)
    # wt = wt[0]
    wt_num = len(wt)
    s = wt[0].shape
    for wn in range(wt_num):
        wt[wn] = wt[wn].view(s[0], s[2], s[3], s[4])

    # bring masks into correct shape
    wtMask = labels.chunk(target_class, dim=1)
    wtMask = list(wtMask)
    s = wtMask[0].shape
    wtMask_label = []
    for wn in range(wt_num):
        wtMask[wn] = wtMask[wn].view(s[0], s[2], s[3], s[4])
        if torch.sum(wtMask[wn]) != 0:
            wtMask_label.append(wn)

    wtMask_num = len(wtMask_label)
    #print(wtMask_num)

    #calculate losses
    wtLoss = []
    wtACLoss = []
    for wn in wtMask_label:#range(wt_num):
        wtLoss_1 = diceLoss(wt[wn], wtMask[wn], nonSquared=nonSquared)# + crossentropy(wt[wn], wtMask[wn])
        wtACLoss_1 = Active_Contour_Loss(wt[wn], wtMask[wn])
        # print('AC:',wtACLoss_1)
        wtLoss.append(wtLoss_1)
        wtACLoss.append(wtACLoss_1 * (1 + wtLoss_1))

    return (np.sum(wtLoss) + np.sum(wtACLoss))/(wtMask_num+1e-7),wtLoss#(wtLoss + tcLoss + etLoss) / 5

# ################organs name###x,  y,  z#####################################
# organs_size = {'Brain Stem': [32, 27, 18],
#                'Eye ball Lens': [21, 21, 9],
#                'Optical Nerve Chiasm Pitutary': [49, 40, 5],
#                'Temporal Lobe': [45, 83, 18],
#                'Parotid glands': [30, 45, 18],
#                'Inner Middle ear': [35, 37, 11],
#                'Mandible T-M Joint': [56, 71, 30],
#                'Spinal cord': [15, 32, 47]}

def NPCDiceLoss(outputs, labels, nonSquared=False):

    target_class = labels.shape[1]

    #bring outputs into correct shape
    wt = outputs.chunk(target_class, dim=1)
    wt = list(wt)
    wt_num = len(wt)
    s = wt[0].shape
    for wn in range(wt_num):
        wt[wn] = wt[wn].view(s[0], s[2], s[3], s[4])

    # bring masks into correct shape
    wtMask = labels.chunk(target_class, dim=1)
    wtMask = list(wtMask)
    s = wtMask[0].shape
    wtMask_label = []
    for wn in range(wt_num):
        wtMask[wn] = wtMask[wn].view(s[0], s[2], s[3], s[4])
        if torch.sum(wtMask[wn]).item() != 0:
            wtMask_label.append(wn)

    wtMask_num = len(wtMask_label)

    #calculate losses
    wtLoss = []
    for wn in wtMask_label:
        wtLoss_1 = diceLoss(wt[wn], wtMask[wn], nonSquared=nonSquared)
        wtLoss.append(wtLoss_1)

    return np.sum(wtLoss) / (wtMask_num + 1e-7), wtLoss


def AllLoss(outputs, labels):
    '''
    :param outputs: the outputs of network
    :param labels: the labels
    :return: final_loss
    '''

    ## Segmentation loss
    loss1 = SegLoss(outputs, labels)

    ## Shape loss
    # loss2 = ShapeLoss(outputs, labels)

    # print("Seg:", loss1.item(), " Shape:", loss2.item())

    ## sum loss
    final_loss = loss1# + loss2 * ((1 - loss1) * (1 - loss1))

    return final_loss

def AllLossReg(outputs, labels):
    '''
    :param outputs: the outputs of network
    :param labels: the labels
    :return: final_loss
    '''

    ## Segmentation loss
    loss1 = SegLoss(outputs[1], labels[1])
    loss2 = SegLoss(outputs[2], labels[2])

    ## Mse loss
    loss3 = MSELoss(outputs[0], labels[0])
    # loss4 = MSELoss(outputs[2], labels[0])

    ## Distance loss
    loss5 = NPCDistLoss(outputs[2], labels[2])
    # loss6 = NPCDistLoss(outputs[3], labels[1])

    ## sum loss
    # final_seg_loss = loss1
    final_mse_loss = loss3
    final_seg_loss = (loss1 + loss2) / 2
    # final_mse_loss = (loss3 + loss4) / 2
    final_dist_loss = loss5

    return final_seg_loss, final_mse_loss, final_dist_loss

def AllLossRegSeg(outputs, labels):
    '''
    :param outputs: the outputs of network
    :param labels: the labels
    :return: final_loss
    '''

    ## Segmentation loss
    loss1 = SegLoss(outputs[1], labels[2])

    ## True Segmentation loss
    loss2 = SegLoss(outputs[2], labels[1])

    ## Mse loss
    loss3 = MSELoss(outputs[0], labels[0])
    # loss4 = MSELoss(outputs[2], labels[0])

    ## Distance loss
    loss5 = NPCDistLoss(outputs[1], labels[2])
    # loss6 = NPCDistLoss(outputs[3], labels[1])

    ## Shape loss
    # loss7 = ShapeLoss(outputs[2], labels[1])

    ## sum loss
    final_seg_loss = loss1
    final_mse_loss = loss3
    final_true_seg_loss = loss2
    # final_seg_loss = (loss1 + loss2) / 2
    # final_mse_loss = (loss3 + loss4) / 2
    final_dist_loss = loss5
    # final_shape_loss = loss7

    return final_seg_loss, final_mse_loss, final_dist_loss, final_true_seg_loss

def MSELoss(outputs, labels):
    # mseloss = torch.nn.MSELoss()
    # mse1 = F.mse_loss(outputs[:, 0, :, :, :], labels[:, 0, :, :, :], reduction='mean')
    # mse2 = F.mse_loss(outputs[:, 1, :, :, :], labels[:, 1, :, :, :], reduction='mean')
    # mse3 = F.mse_loss(outputs[:, 2, :, :, :], labels[:, 2, :, :, :], reduction='mean')
    # mse = mse1 + mse2 + mse3
    # mse = mse / 3
    mse = F.mse_loss(outputs, labels, reduction='mean')
    return mse

def SegLoss(outputs, labels):
    ## Loss for Segmentation task
    target_class = labels.shape[1]

    # bring outputs into correct shape
    wt = outputs.chunk(target_class, dim=1)
    wt = list(wt)
    wt_num = len(wt)
    s = wt[0].shape
    for wn in range(wt_num):
        wt[wn] = wt[wn].view(s[0], s[2], s[3], s[4])

    # bring masks into correct shape
    wtMask = labels.chunk(target_class, dim=1)
    wtMask = list(wtMask)
    s = wtMask[0].shape
    wtMask_label = []
    for wn in range(wt_num):
        wtMask[wn] = wtMask[wn].view(s[0], s[2], s[3], s[4])
        if torch.sum(wtMask[wn]).item() != 0:
            wtMask_label.append(wn)

    wtMask_num = len(wtMask_label)

    # calculate losses
    wtLoss = []
    for wn in wtMask_label:
        wtLoss_1 = diceLoss(wt[wn], wtMask[wn])
        wtLoss.append(wtLoss_1)

    return np.sum(wtLoss) / (wtMask_num + 1e-7)

def ShapeLoss(outputs, labels):
    ## Loss for Segmentation task
    target_class = labels.shape[1]

    # bring outputs into correct shape
    wt = outputs.chunk(target_class, dim=1)
    wt = list(wt)
    wt_num = len(wt)
    s = wt[0].shape
    for wn in range(wt_num):
        wt[wn] = wt[wn].view(s[0], s[2], s[3], s[4])

    # bring masks into correct shape
    wtMask = labels.chunk(target_class, dim=1)
    wtMask = list(wtMask)
    s = wtMask[0].shape
    wtMask_label = []
    for wn in range(wt_num):
        wtMask[wn] = wtMask[wn].view(s[0], s[2], s[3], s[4])
        if torch.sum(wtMask[wn]).item() != 0:
            wtMask_label.append(wn)

    wtMask_num = len(wtMask_label)

    # calculate losses
    wtLoss = []
    for wn in wtMask_label:
        wtLoss_1 = HuMomentLoss(wt[wn], wtMask[wn])
        wtLoss.append(wtLoss_1)

    return np.sum(wtLoss) / (wtMask_num + 1e-7)

def HuMomentLoss(outputs, labels):
    ## Loss for Segmentation task

    outputs = torch.sum(outputs, dim=-1) / labels.shape[3]
    labels = torch.sum(labels, dim=-1) / labels.shape[3]

    Hu_outputs = HuMoment(outputs[0, :, :])
    Hu_labels = HuMoment(labels[0, :, :])

    HuLoss = []
    for i in range(len(Hu_labels)):
        HuLoss.append(torch.abs(Hu_outputs[i] - Hu_labels[i]))
    HuLoss = np.sum(HuLoss) / 7

    # HuLoss = torch.zeros([1]).cuda()
    # z_count = 0
    # for z in range(labels.shape[3]):
    #     if torch.sum(labels[0, :, :, z]) == 0:
    #         continue
    #     Hu_outputs = HuMoment(outputs[0, :, :, z])
    #     Hu_labels = HuMoment(labels[0, :, :, z])
    #     HuLoss_z = []
    #     for i in range(len(Hu_labels)):
    #         HuLoss_z.append(torch.abs(Hu_outputs[i] - Hu_labels[i]))
    #     HuLoss_z = np.sum(HuLoss_z) / 7
    #     HuLoss += HuLoss_z
    #     z_count += 1
    # HuLoss = HuLoss / z_count
    return HuLoss




def HuMoment(map):
    if map.is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    matrix_index = torch.ones([map.shape[0], map.shape[1], 2])
    matrix_index = matrix_index.to(device)

    matrix_index[:, :, 0] = torch.mul(matrix_index[:, :, 0],
                                      torch.reshape(torch.range(0, map.shape[0] - 1), [map.shape[0], 1]).to(device))
    matrix_index[:, :, 1] = torch.mul(matrix_index[:, :, 1],
                                      torch.reshape(torch.range(0, map.shape[1] - 1), [1, map.shape[1]]).to(device))



    m00 = torch.sum(map)
    m10 = torch.sum(map * matrix_index[:, :, 0])
    m01 = torch.sum(map * matrix_index[:, :, 1])

    x0 = m10 / (m00 + 1e-7)
    y0 = m01 / (m00 + 1e-7)

    u00 = m00
    n20 = torch.sum(map * ((matrix_index[:, :, 0] - x0) ** 2)) / (u00 + 1e-7)
    n02 = torch.sum(map * ((matrix_index[:, :, 1] - y0) ** 2)) / (u00 + 1e-7)
    n11 = torch.sum(map * (matrix_index[:, :, 1] - y0) * (matrix_index[:, :, 0] - x0)) / (u00 + 1e-7)
    n12 = torch.sum(map * (matrix_index[:, :, 0] - x0) * ((matrix_index[:, :, 1] - y0) ** 2)) / ((u00 ** 1.5) + 1e-7)
    n21 = torch.sum(map * ((matrix_index[:, :, 0] - x0) ** 2) * (matrix_index[:, :, 1] - y0)) / ((u00 ** 1.5) + 1e-7)
    n30 = torch.sum(map * ((matrix_index[:, :, 0] - x0) ** 3)) / ((u00 ** 1.5) + 1e-7)
    n03 = torch.sum(map * ((matrix_index[:, :, 1] - y0) ** 3)) / ((u00 ** 1.5) + 1e-7)

    h1 = n20 + n02
    h2 = (n20 - n02) ** 2 + 4 * (n11 ** 2)
    h3 = (n20 - 3*n12) ** 2 + 3 * ((n21 - n03) ** 2)
    h4 = (n30 + n12) ** 2 + (n21 + n03) ** 2
    h5 = (n30 + 3 * n12) * (n30 + n12) * ((n30 + n12)**2 - 3*((n21 + n03) ** 2)) + \
         (3*n21 - n03) * (n21 + n03) * (3 * ((n30 + n21) ** 2) - (n21 + n03) ** 2)
    h6 = (n20 - n02) * ((n30 + n21) ** 2 - (n21 + n03) ** 2) + 4 * n11 * (n30 + n12) * (n21 + n03)
    h7 = (3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) - \
         (n30 - n12) * (n21 + n03) * (3 * (n30 + n21) ** 2 - (n21 + n03) ** 2)

    h1 = torch.sign(h1) * torch.log10(torch.abs(h1)+1e-7)
    h2 = torch.sign(h2) * torch.log10(torch.abs(h2)+1e-7)
    h3 = torch.sign(h3) * torch.log10(torch.abs(h3)+1e-7)
    h4 = torch.sign(h4) * torch.log10(torch.abs(h4)+1e-7)
    h5 = torch.sign(h5) * torch.log10(torch.abs(h5)+1e-7)
    h6 = torch.sign(h6) * torch.log10(torch.abs(h6)+1e-7)
    h7 = torch.sign(h7) * torch.log10(torch.abs(h7)+1e-7)

    HuM = [h1, h2, h3, h4, h5, h6, h7]

    return HuM


def NPCRegionDiceLoss(outputs, labels, nonSquared=False, target_class=22):

    target_class = labels.shape[1]

    # bring outputs into correct shape
    wt = outputs.chunk(target_class, dim=1)
    wt = list(wt)
    # wt = wt[0]
    wt_num = len(wt)
    s = wt[0].shape
    for wn in range(wt_num):
        wt[wn] = wt[wn].view(s[0], s[2], s[3], s[4])

    # bring masks into correct shape
    wtMask = labels.chunk(target_class, dim=1)
    wtMask = list(wtMask)
    s = wtMask[0].shape
    wtMask_label = []
    for wn in range(wt_num):
        wtMask[wn] = wtMask[wn].view(s[0], s[2], s[3], s[4])
        # print(torch.sum(wtMask[wn]).item())
        if torch.sum(wtMask[wn]).item() != 0:
            wtMask_label.append(wn)

    wtMask_num = len(wtMask_label)
    # print(wtMask_num)

    # calculate losses
    wtLoss = []
    wtLoss_region = []
    for wn in wtMask_label:  # range(wt_num):
        wtLoss_1 = diceLoss(wt[wn], wtMask[wn], nonSquared=nonSquared)  # + crossentropy(wt[wn], wtMask[wn])
        wtLoss_region_1 = Region_Loss(wt[wn], wtMask[wn])
        wtLoss.append(wtLoss_1)
        wtLoss_region.append(wtLoss_region_1)

        print('Dice Loss:', (np.sum(wtLoss) / (wtMask_num + 1e-7)).item(),
              ' Region Loss:', (np.sum(wtLoss_region) / (wtMask_num + 1e-7)).item())

    return (np.sum(wtLoss)*0.5 + np.sum(wtLoss_region)*0.5) / (wtMask_num + 1e-7), wtLoss

    # return (np.sum(wtLoss) + np.sum(wtLoss_dis) * 0.1)/(wtMask_num+1e-7),wtLoss

def NPCDistLoss(outputs, labels):

    target_class = labels.shape[1]

    #bring outputs into correct shape
    wt = outputs.chunk(target_class, dim=1)
    wt = list(wt)

    wt_num = len(wt)
    s = wt[0].shape
    for wn in range(wt_num):
        wt[wn] = wt[wn].view(s[0], s[2], s[3], s[4])

    # bring masks into correct shape
    wtMask = labels.chunk(target_class, dim=1)
    wtMask = list(wtMask)
    s = wtMask[0].shape
    wtMask_label = []
    for wn in range(wt_num):
        wtMask[wn] = wtMask[wn].view(s[0], s[2], s[3], s[4])
        # print(torch.sum(wtMask[wn]).item())
        if torch.sum(wtMask[wn]).item() != 0:
            wtMask_label.append(wn)

    wtMask_num = len(wtMask_label)

    #calculate losses
    wtLoss_dis = []
    for wn in wtMask_label:#range(wt_num):
        wtLoss_dis_1 = LocationLoss(wt[wn], wtMask[wn])
        wtLoss_dis.append(wtLoss_dis_1)

    return np.sum(wtLoss_dis) / (wtMask_num + 1e-7)

def NPCMultiDiceLoss(outputs, labels, nonSquared=False, target_class = 1):

    target_class = labels.shape[1]

    #bring outputs into correct shape
    wt = outputs.chunk(target_class, dim=1)
    wt = list(wt)
    # wt = wt[0]
    wt_num = len(wt)
    s = wt[0].shape
    for wn in range(wt_num):
        wt[wn] = wt[wn].view(s[0], s[2], s[3], s[4])

    # bring masks into correct shape
    wtMask = labels.chunk(target_class, dim=1)
    wtMask = list(wtMask)
    s = wtMask[0].shape
    wtMask_label = []
    for wn in range(wt_num):
        wtMask[wn] = wtMask[wn].view(s[0], s[2], s[3], s[4])
        if torch.sum(wtMask[wn]) != 0:
            wtMask_label.append(wn)

    wtMask_num = len(wtMask_label)

    #calculate losses
    wtLoss = []
    for wn in wtMask_label:#range(wt_num):
        wtLoss_1 = diceLoss(wt[wn], wtMask[wn], nonSquared=nonSquared) + crossentropy(wt[wn], wtMask[wn])
        wtLoss.append(wtLoss_1)

    return np.sum(wtLoss)/(wtMask_num+1e-7),wtLoss#(wtLoss + tcLoss + etLoss) / 5

def NPCSpinalLoss(outputs, labels, nonSquared=False, target_class = 1):

    #bring outputs into correct shape
    wt = outputs.chunk(target_class, dim=1)
    wt = list(wt)
    # wt = wt[0]
    wt_num = len(wt)
    s = wt[0].shape
    for wn in range(wt_num):
        wt[wn] = wt[wn].view(s[0], s[2], s[3], s[4])

    # bring masks into correct shape
    wtMask = labels.chunk(target_class, dim=1)
    wtMask = list(wtMask)
    s = wtMask[0].shape
    wtMask_label = []
    for wn in range(wt_num):
        wtMask[wn] = wtMask[wn].view(s[0], s[2], s[3], s[4])
        if torch.sum(wtMask[wn]) != 0:
            wtMask_label.append(wn)

    wtMask_num = len(wtMask_label)

    #calculate losses
    wtLoss = []
    for wn in wtMask_label:#range(wt_num):
        wtLoss_1 = diceLoss(wt[wn], wtMask[wn], nonSquared=nonSquared) + crossentropy(wt[wn], wtMask[wn])
        wtLoss.append(wtLoss_1)

    return np.sum(wtLoss)/(wtMask_num+1e-7),wtLoss#(wtLoss + tcLoss + etLoss) / 5


def NPCSmallLoss(outputs, labels, nonSquared=False, target_class = 5):

    #bring outputs into correct shape
    wt = outputs.chunk(target_class, dim=1)
    wt = list(wt)
    # wt = wt[0]
    wt_num = len(wt)
    s = wt[0].shape
    for wn in range(wt_num):
        wt[wn] = wt[wn].view(s[0], s[2], s[3], s[4])

    # bring masks into correct shape
    wtMask = labels.chunk(target_class, dim=1)
    wtMask = list(wtMask)
    s = wtMask[0].shape
    wtMask_label = []
    for wn in range(wt_num):
        wtMask[wn] = wtMask[wn].view(s[0], s[2], s[3], s[4])
        if torch.sum(wtMask[wn]) != 0:
            wtMask_label.append(wn)

    wtMask_num = len(wtMask_label)

    #calculate losses
    wtLoss = []
    for wn in range(wt_num):
        wtLoss_1 = diceLoss(wt[wn], wtMask[wn], nonSquared=nonSquared) + crossentropy(wt[wn], wtMask[wn])
        wtLoss.append(wtLoss_1)

    return np.sum(wtLoss)/(wt_num+1e-7),wtLoss#(wtLoss + tcLoss + etLoss) / 5

def EuclideanLoss(outputs, labels):
    cha = outputs - labels
    cha = torch.mul(cha,cha)
    cha = torch.sum(cha)
    EL = torch.sqrt(cha)
    return EL

def NPCGTVLoss(outputs, labels, nonSquared=False, target_class = 1):

    #bring outputs into correct shape
    wt = outputs.chunk(target_class, dim=1)
    wt = list(wt)
    # wt = wt[0]
    wt_num = len(wt)
    s = wt[0].shape
    for wn in range(wt_num):
        wt[wn] = wt[wn].view(s[0], s[2], s[3], s[4])

    # bring masks into correct shape
    wtMask = labels.chunk(target_class, dim=1)
    wtMask = list(wtMask)
    s = wtMask[0].shape
    wtMask_label = []
    for wn in range(wt_num):
        wtMask[wn] = wtMask[wn].view(s[0], s[2], s[3], s[4])
        if torch.sum(wtMask[wn]) != 0:
            wtMask_label.append(wn)

    wtMask_num = len(wtMask_label)

    #calculate losses
    wtLoss = []
    for wn in wtMask_label:#range(wt_num):
        wtLoss_1 = diceLoss(wt[wn], wtMask[wn], nonSquared=nonSquared)# + (0.1 * EuclideanLoss(wt[wn], wtMask[wn]))
        wtLoss.append(wtLoss_1)

    return np.sum(wtLoss)/(wtMask_num+1e-7),wtLoss#(wtLoss + tcLoss + etLoss) / 5

def NPCWeightDiceLoss(outputs, labels, nonSquared=False, target_class = 13):

    #bring outputs into correct shape
    wt = outputs.chunk(target_class, dim=1)
    wt = list(wt)
    # wt = wt[0]
    wt_num = len(wt)
    s = wt[0].shape
    for wn in range(wt_num):
        wt[wn] = wt[wn].view(s[0], s[2], s[3], s[4])

    # bring masks into correct shape
    wtMask = labels.chunk(target_class, dim=1)
    wtMask = list(wtMask)
    s = wtMask[0].shape
    wtMask_label = []
    for wn in range(wt_num):
        wtMask[wn] = wtMask[wn].view(s[0], s[2], s[3], s[4])
        # if torch.sum(wtMask[wn]) != 0:
        #     wtMask_label.append(wn)

    wtMask_num = len(wtMask_label)

    #calculate losses
    wtLoss = []
    for wn in range(wt_num):
        wtLoss_1 = diceLoss(wt[wn], wtMask[wn], nonSquared=nonSquared)
        wtLossWeight = 0.4 + (1 - dice(wt[wn], wtMask[wn]))
        wtLoss.append(wtLoss_1 * wtLossWeight)

    return np.sum(wtLoss)/(wt_num+1e-7),wtLoss#(wtLoss + tcLoss + etLoss) / 5

DisMean = [[-3, 73, -28],[-3, 73, 28],[-3, 79, -28],[-3,79,28],[-3,52,-20],[-3,52,20],[-4,30,0],
           [-6,1,-38],[-5,2,38],[-2,26,0],[16,12,-53],[16,11,53],[3,4,-31],[3,3,31],[5,-3,-46],
           [5,-3,45],[5,18,-46],[6,17,47],[22,49,-30],[22,49,29]]
DisStd = [[3.1,6.1,3.3],[3.2,6.1,3.7],[3.6,6.6,3.4],[3.6,6.5,3.9],[2.3,4.4,2.4],[2.3,4.5,2.8],
          [1.6,3.6,1.2],[1.4,4.9,3.4],[1.4,4.7,3.3],[1.4,2.6,1.2],[1.7,5.7,5.0],[1.5,5.1,4.6],
          [0.8,2.1,3.0],[0.9,2.1,2.9],[1.5,3.0,3.8],[1.3,3.1,3.6],[1.3,3.7,4.2],[1.4,3.2,4.1],
          [2.4,8.5,3.6],[2.3,8.1,3.1]]

# DisMean = [[-3, 73, 0],[-3, 79, 0],[-3,52,0],[-4,30,0],[-6,1,0],[-2,26,0],[16,12,0],[3,4,0],[5,-3,0],[5,18,0],[22,49,0]]
# DisStd = [[3.1,6.1,3.3],[3.6,6.5,3.9],[2.3,4.5,2.8], [1.6,3.6,1.2],[1.4,4.7,3.3],[1.4,2.6,1.2],[1.5,5.1,4.6],[0.9,2.1,2.9],[1.5,3.0,3.8],[1.4,3.2,4.1],[2.4,8.5,3.6]]

def NPCDiceLocationLoss(outputs, labels, nonSquared=False, target_class = 13):

    #bring outputs into correct shape
    wt = outputs.chunk(target_class, dim=1)
    wt = list(wt)
    # wt = wt[0]
    wt_num = len(wt)
    s = wt[0].shape
    for wn in range(wt_num):
        wt[wn] = wt[wn].view(s[0], s[2], s[3], s[4])
    BrainStemCentroid = LocationCentroid(wt[0])

    # bring masks into correct shape
    wtMask = labels.chunk(target_class, dim=1)
    wtMask = list(wtMask)
    s = wtMask[0].shape
    wtMask_label = []
    for wn in range(wt_num):
        wtMask[wn] = wtMask[wn].view(s[0], s[2], s[3], s[4])
        if torch.sum(wtMask[wn]) != 0:
            wtMask_label.append(wn)

    BrainStemMaskCentroid = LocationCentroid(wtMask[0])

    wtMask_num = len(wtMask_label)

    #calculate losses
    wtLoss = []
    wtLocationLoss = []
    for wn in wtMask_label:
        if wn == 0:
            wtLocationLoss_1 = torch.sum(torch.abs(BrainStemMaskCentroid - BrainStemCentroid))
        else:
            wtLocationLoss_1 = LocationLoss(wt[wn], BrainStemCentroid,DisMean[wn-1],DisStd[wn-1])
        wtLocationLoss_1 = wtLocationLoss_1 * 0.1
        wtLoss_1 = diceLoss(wt[wn], wtMask[wn], nonSquared=nonSquared)
        wtLoss.append(wtLoss_1)
        wtLocationLoss.append(wtLocationLoss_1)
    print(np.sum(wtLocationLoss)/(wtMask_num+1e-7))
    return (np.sum(wtLoss)/(wtMask_num+1e-7)) + (np.sum(wtLocationLoss)/(wtMask_num+1e-7)),wtLoss#(wtLoss + tcLoss + etLoss) / 5


def NPCDice2DLoss(outputs, labels, nonSquared=False, target_class = 13):

    #bring outputs into correct shape
    wt = outputs.chunk(target_class, dim=1)
    wt = list(wt)
    # wt = wt[0]
    wt_num = len(wt)
    s = wt[0].shape
    for wn in range(wt_num):
        wt[wn] = wt[wn].view(s[0], s[2], s[3])

    # bring masks into correct shape
    wtMask = labels.chunk(target_class, dim=1)
    wtMask = list(wtMask)
    s = wtMask[0].shape
    wtMask_label = []
    for wn in range(wt_num):
        wtMask[wn] = wtMask[wn].view(s[0], s[2], s[3])
        if torch.sum(wtMask[wn]) != 0:
            wtMask_label.append(wn)

    wtMask_num = len(wtMask_label)

    #calculate losses
    wtLoss = []
    for wn in wtMask_label:
        wtLoss_1 = tverskydice2DLoss(wt[wn], wtMask[wn], nonSquared=nonSquared)
        wtLoss.append(wtLoss_1)

    return np.sum(wtLoss)/wtMask_num,wtLoss#(wtLoss + tcLoss + etLoss) / 5


def NPCDiceCELoss(outputs, labels, nonSquared=False, target_class = 13):

    #bring outputs into correct shape
    wt = outputs.chunk(target_class, dim=1)
    wt = list(wt)
    # wt = wt[0]
    wt_num = len(wt)
    s = wt[0].shape
    for wn in range(wt_num):
        wt[wn] = wt[wn].view(s[0], s[2], s[3], s[4])

    # bring masks into correct shape
    wtMask = labels.chunk(target_class, dim=1)
    wtMask = list(wtMask)
    s = wtMask[0].shape
    for wn in range(wt_num):
        wtMask[wn] = wtMask[wn].view(s[0], s[2], s[3], s[4])

    #calculate losses
    wtLoss = []
    for wn in range(wt_num):
        wtLoss_1 = ceadddiceLoss(wt[wn], wtMask[wn], nonSquared=nonSquared)
        wtLoss.append(wtLoss_1)

    return np.sum(wtLoss)/target_class,wtLoss#(wtLoss + tcLoss + etLoss) / 5


def NPCFocalDiceLoss(outputs, labels, nonSquared=False, target_class = 4):

    #bring outputs into correct shape
    wt = outputs.chunk(target_class, dim=1)
    wt = list(wt)
    # wt = wt[0]
    wt_num = len(wt)
    s = wt[0].shape
    for wn in range(wt_num):
        wt[wn] = wt[wn].view(s[0], s[2], s[3], s[4])

    # bring masks into correct shape
    wtMask = labels.chunk(target_class, dim=1)
    wtMask = list(wtMask)
    s = wtMask[0].shape
    for wn in range(wt_num):
        wtMask[wn] = wtMask[wn].view(s[0], s[2], s[3], s[4])

    #calculate losses
    wtLoss = []
    for wn in range(wt_num):
        wtLoss_1 = focaladddiceLoss(wt[wn], wtMask[wn], nonSquared=nonSquared)
        wtLoss.append(wtLoss_1)
    # wt, tc, et = outputs.chunk(3, dim=1)
    # s = wt.shape
    # wt = wt.view(s[0], s[2], s[3], s[4])
    # tc = tc.view(s[0], s[2], s[3], s[4])
    # et = et.view(s[0], s[2], s[3], s[4])
    #
    # # bring masks into correct shape
    # wtMask, tcMask, etMask = labels.chunk(3, dim=1)
    # s = wtMask.shape
    # wtMask = wtMask.view(s[0], s[2], s[3], s[4])
    # tcMask = tcMask.view(s[0], s[2], s[3], s[4])
    # etMask = etMask.view(s[0], s[2], s[3], s[4])
    #
    # #calculate losses
    # wtLoss = diceLoss(wt, wtMask, nonSquared=nonSquared)
    # tcLoss = diceLoss(tc, tcMask, nonSquared=nonSquared)
    # etLoss = diceLoss(et, etMask, nonSquared=nonSquared)
    return np.sum(wtLoss)/target_class,wtLoss#(wtLoss + tcLoss + etLoss) / 5

def NPCWeightedDiceLoss(outputs, labels, nonSquared=False):

    #bring outputs into correct shape
    wt = outputs.chunk(12, dim=1)
    wt = list(wt)
    # wt = wt[0]
    wt_num = len(wt)
    s = wt[0].shape
    for wn in range(wt_num):
        wt[wn] = wt[wn].view(s[0], s[2], s[3], s[4])

    # bring masks into correct shape
    wtMask = labels.chunk(12, dim=1)
    wtMask = list(wtMask)
    s = wtMask[0].shape
    for wn in range(wt_num):
        wtMask[wn] = wtMask[wn].view(s[0], s[2], s[3], s[4])

    weight_list = [1, 1, 1.5, 2, 1, 1.5, 1, 1, 1.2, 1.2, 1.2, 1]

    #calculate losses
    wtLoss = []
    for wn in range(wt_num):
        wtLoss_1 = weight_list[wn]*tverskydiceLoss(wt[wn], wtMask[wn], nonSquared=nonSquared)
        wtLoss.append(wtLoss_1)

    return np.sum(wtLoss)/12,wtLoss#(wtLoss + tcLoss + etLoss) / 5


def NPCDiceLossOriginal5(outputs, labels, nonSquared=False):
    outputList = list(outputs.chunk(5, dim=1))
    labelsList = list(labels.chunk(5, dim=1))
    totalLoss = 0
    for pred, target in zip(outputList, labelsList):
        totalLoss = totalLoss + diceLoss(pred, target, nonSquared=nonSquared)
    return totalLoss


def sensitivity(pred, target):
    predBin = (pred >= 0.5).float()
    intersection = (predBin * target).sum()
    allPositive = target.sum()

    # special case for zero positives
    if allPositive == 0:
        return 1.0
    return (intersection / allPositive).item()

def specificity(pred, target):
    predBinInv = (pred <= 0.5).float()
    targetInv = (target == 0).float()
    intersection = (predBinInv * targetInv).sum()
    allNegative = targetInv.sum()
    return (intersection / allNegative).item()

def getHd95(pred, target):
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    if np.count_nonzero(pred) > 0 and np.count_nonzero(target):
        surDist1 = medpyMetrics.__surface_distances(pred, target)
        surDist2 = medpyMetrics.__surface_distances(target, pred)
        hd95 = np.percentile(np.hstack((surDist1, surDist2)), 95)
        return hd95
    else:
        # Edge cases that medpy cannot handle
        return -1

def getWTMask(labels):
    return (labels != 0).float()

def getTCMask(labels):
    return ((labels != 0) * (labels != 2)).float() #We use multiplication as AND

def getETMask(labels):
    return (labels == 4).float()
