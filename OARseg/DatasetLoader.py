import torch
import torch.utils.data
import numpy as np
import os
import sys
import dataProcessing.augmentation as aug
import dataProcessing.utils as utils
import copy

class OARDataset(torch.utils.data.Dataset):
    #mode must be trian, test or val

    def __init__(self, filePath, expConfig, mask_num=None, mode="train", hasMasks=True, AutoChangeDim=True):
        super(OARDataset, self).__init__()
        self.filePath = filePath
        self.mode = mode
        self.hasMasks = hasMasks
        self.mask_num = mask_num
        self.auto_dim = AutoChangeDim

        #augmentation settings
        self.nnAugmentation = expConfig.NN_AUGMENTATION
        self.softAugmentation = expConfig.SOFT_AUGMENTATION
        self.doRotate = expConfig.DO_ROTATE
        self.rotDegrees = expConfig.ROT_DEGREES
        self.doScale = expConfig.DO_SCALE
        self.scaleFactor = expConfig.SCALE_FACTOR
        self.doFlip = expConfig.DO_FLIP
        self.doElasticAug = expConfig.DO_ELASTIC_AUG
        self.sigma = expConfig.SIGMA
        self.doIntensityShift = expConfig.DO_INTENSITY_SHIFT
        self.maxIntensityShift = expConfig.MAX_INTENSITY_SHIFT

    def __getitem__(self, index):

        ## lazily open file
        self.readDataList()

        ## get patient id
        patient_id = self.file[index]
        patient_id = patient_id.split('_')[0]

        ## get data path
        image_path = os.path.join(self.filePath, self.file[index] + '_data.nii.gz')
        local_result_path = os.path.join(self.filePath, self.file[index] + '_label.nii.gz')

        ## read data
        image = utils.read_nii_image(image_path)
        local_result = utils.read_nii_image(local_result_path)

        ## check the dimension number
        img_shape = image.shape
        try:
            if len(img_shape) > 3 or len(img_shape) < 2: raise
        except RuntimeError:
            print("\033[31m" + "Image shape should be [2] for 2D or 3 for 3D! Wrong Dimension:", img_shape)
            sys.exit(1)

        if self.hasMasks:
            mask_path = os.path.join(self.filePath, self.file[index] + '_label.nii.gz')
            labels = utils.read_nii_image(mask_path)

            ## shape matching
            lbs_shape = labels.shape
            try:
                if img_shape != lbs_shape: raise
            except RuntimeError:
                print("\033[31m" + "Image shape and label shape are not match ", img_shape, ' vs. ', lbs_shape)
                sys.exit(1)

        ## transpose z axis to the last dimension
        if self.auto_dim:
            min_dim = np.argmin(img_shape)
            dim_num = int(len(img_shape) - 1)
            image.swapaxes(min_dim, dim_num)
            image = image.astype('float32')
            local_result.swapaxes(min_dim, dim_num)
            local_result = local_result.astype('float32')
            if self.hasMasks:
                labels.swapaxes(min_dim, dim_num)
                labels = labels.astype('float32')
        else:
            image = np.transpose(image, [1, 2, 0])
            image = image.astype('float32')
            local_result = np.transpose(local_result, [1, 2, 0])
            local_result = local_result.astype('float32')
            if self.hasMasks:
                labels = np.transpose(labels, [1, 2, 0])
                labels = labels.astype('float32')


        ## Additional operation by Bin Huang
        image[image < -900] = -900
        ## image normalization
        image = self.normalise_image(image, min_intensity=-900)

        local_result = self._toEvaluationOneHot(local_result, self.mask_num)

        ## add a channel axis
        if len(img_shape) == image.ndim: image = np.expand_dims(image, -1)

        #Prepare data
        if self.hasMasks:
            ## labels to multichannels
            labels = self._toEvaluationOneHot(labels, self.mask_num)
            if labels.ndim < 4:
                labels = np.expand_dims(labels, 3)
            defaultLabelValues = np.asarray([0], dtype=np.float32)



        #augment data
        if self.mode == "train":
            image, labels, local_result = aug.augment3DImageEdge(image,
                                                                 labels,
                                                                 local_result,
                                                                 defaultLabelValues,
                                                                 self.nnAugmentation,
                                                                 self.doRotate,
                                                                 self.rotDegrees,
                                                                 self.doScale,
                                                                 self.scaleFactor,
                                                                 self.doFlip,
                                                                 self.doElasticAug,
                                                                 self.sigma,
                                                                 self.doIntensityShift,
                                                                 self.maxIntensityShift)

        image = np.transpose(image, (3, 0, 1, 2))  # bring into NCWH format
        local_result = np.transpose(local_result, (3, 0, 1, 2))
        if self.hasMasks: labels = np.transpose(labels, (3, 0, 1, 2))  # bring into NCWH format

        ## to tensor
        image = torch.from_numpy(image)
        local_result = torch.from_numpy(local_result)
        if self.hasMasks:
            labels = torch.from_numpy(labels)

        ## return index
        if self.hasMasks:
            return [image, local_result], str(patient_id), [labels]
        else:
            return [image, local_result], str(patient_id)

    def __len__(self):
        #Read Data list
        self.readDataList()
        return len(self.file)

    def readDataList(self):
        filelist = os.listdir(self.filePath)
        self.file = []
        for fl in filelist:
            if 'data' in fl:
                fl_split = fl.split('_data')
                fl_num = fl_split[0]
                self.file.append(fl_num)

    def _toEvaluationOneHot(self, labels, mask_num):
        shape = labels.shape

        ## create labels matrix with channels
        if mask_num:
            out = np.zeros([shape[0], shape[1], shape[2], mask_num], dtype=np.float32)
        else:
            mask_num = int(np.max(labels))
            out = np.zeros([shape[0], shape[1], shape[2], mask_num], dtype=np.float32)

        ## get the labels
        for i in range(mask_num):
            out[:, :, :, i] = (labels == i + 1)
        return out

    def normalise_image(self, image, min_intensity=-1024):
        '''
        standardize based on nonzero pixels
        '''
        m = np.nanmean(np.where(image <= min_intensity, np.nan, image))
        s = np.nanstd(np.where(image <= min_intensity, np.nan, image))
        normalized = np.divide((image - m), s)
        image = np.where(image == min_intensity, 0, normalized)
        return image

    def min_max_normalise_image(self,image,min_intensity = -1024):
        '''
        standardize based on nonzero pixels
        '''
        max_intens = np.nanmax(np.where(image <= min_intensity, np.nan, image), axis=(0, 1, 2)).astype(np.float32)
        min_intens = np.nanmin(np.where(image <= min_intensity, np.nan, image), axis=(0, 1, 2)).astype(np.float32)
        normalized = np.divide((image - min_intens), (max_intens - min_intens))
        image = np.where(image == min_intensity, 0, normalized)
        return image

    def normalise_image_bin(self, image, mean, std):
        image = (image - mean) / std
        return image

    def ChangetoMultiChannels(self, image):
        img_bg = copy.deepcopy(image)
        img_bg[img_bg != -1000] = 1
        img_bg[img_bg == -1000] = 0
        img_c1 = copy.deepcopy(image)
        img_c1 = self.normalise_image_bin(img_c1, 117.7, 355.8)
        img_c1 = img_c1 * img_bg

        img_c2 = copy.deepcopy(image)
        img_c2[img_c2 < -450] = -450
        img_c2[img_c2 > 1050] = 1050
        img_c2 = self.normalise_image_bin(img_c2, 118.9, 302.1)
        img_c2 = img_c2 * img_bg

        img_c3 = copy.deepcopy(image)
        img_c3[img_c3 < -200] = -200
        img_c3[img_c3 > 300] = 300
        img_c3 = self.normalise_image_bin(img_c3, 61.0, 127.5)
        img_c3 = img_c3 * img_bg
        img_dat = np.stack((img_c1, img_c2, img_c3), 3)
        return img_dat

if __name__ == '__main__':
    #load data
    import matplotlib.pyplot as plt
    import experiments.noNewNet_Bin_OARs_endtoend as expConfig
    import systemsetup
    trainset = OARDataset(systemsetup.DATA_PATH, expConfig, mode="train", mask_num = 9, hasMasks=True, AutoChangeDim=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=expConfig.BATCH_SIZE, shuffle=True, pin_memory=False, num_workers=expConfig.DATASET_WORKERS)

    for ii, sample in enumerate(trainloader):
        img = sample[0].numpy()
        gt = sample[2].numpy()

        print(sample[1])

        for i in range(96):
            if np.sum(gt[0,2,:,:,i]) >0:
                plt.figure()
                plt.title('display')
                plt.subplot(211)
                plt.imshow(img[0, 0, :, :,i])
                plt.subplot(212)
                plt.imshow(gt[0, 2, :, :,i])
                plt.show(block=True)
                # print('\n')
