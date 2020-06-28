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
        self.local_result_path = "/media/root/01456da2-1f1d-4b67-810e-b9cd3341133d/2019_MICCAI_challenge_NPC/predictbase/202006150950_MICCAI2015OAR_registration_alltemplate_globalmaxpooling_morechannels_distloss"
        self.template_path = "/media/root/01456da2-1f1d-4b67-810e-b9cd3341133d/NPC_MICCAI_2015_original_data/HaN_2015_crop/train_all_headonly_affine_template_seg"

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
        # print(patient_id)
        ## get data path
        image_path = os.path.join(self.filePath, self.file[index] + '_data.nii.gz')
        local_result_path = os.path.join(self.filePath, self.file[index] + '_label.nii.gz')
        template_path = os.path.join(self.template_path, self.file[index] + '_label.nii.gz')

        ## read data
        image = utils.read_nii_image(image_path)
        local_result = utils.read_nii_image(local_result_path)
        template = utils.read_nii_image(template_path)

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
            template.swapaxes(min_dim, dim_num)
            template = template.astype('float32')
            if self.hasMasks:
                labels.swapaxes(min_dim, dim_num)
                labels = labels.astype('float32')
        else:
            image = np.transpose(image, [1, 2, 0])
            image = image.astype('float32')
            local_result = np.transpose(local_result, [1, 2, 0])
            local_result = local_result.astype('float32')
            template = np.transpose(template, [1, 2, 0])
            template = template.astype('float32')
            if self.hasMasks:
                labels = np.transpose(labels, [1, 2, 0])
                labels = labels.astype('float32')


        ## Additional operation by Bin Huang
        image[image < -900] = -900
        ## image normalization
        # image = self.normalise_image(image, min_intensity=-900)
        image = self.ChangetoMultiChannels(image)

        local_result = self._toEvaluationOneHot(local_result, self.mask_num)
        template = self._toEvaluationOneHot(template, self.mask_num)

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
            image, labels, local_result, template = aug.augment3DImageEdge(image,
                                                                           labels,
                                                                           local_result,
                                                                           template,
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
        template = np.transpose(template, (3, 0, 1, 2))
        if self.hasMasks: labels = np.transpose(labels, (3, 0, 1, 2))  # bring into NCWH format

        ## to tensor
        image = torch.from_numpy(image)
        local_result = torch.from_numpy(local_result)
        template = torch.from_numpy(template)
        if self.hasMasks:
            labels = torch.from_numpy(labels)

        ## return index
        if self.hasMasks:
            return [image, local_result, template], str(patient_id), [labels]
        else:
            return [image, local_result, template], str(patient_id)

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
        img_bg[img_bg < -900] = -900
        img_bg[img_bg > 1500] = 1500
        img_c1 = copy.deepcopy(image)
        img_c1 = self.normalise_image(img_c1, min_intensity=-900)

        img_c2 = copy.deepcopy(image)
        img_c2[img_c2 < -450] = -450
        img_c2[img_c2 > 1050] = 1050
        img_c2 = self.normalise_image(img_c2, min_intensity=-450)

        img_c3 = copy.deepcopy(image)
        img_c3[img_c3 < -200] = -200
        img_c3[img_c3 > 300] = 300
        img_c3 = self.normalise_image(img_c3, min_intensity=-200)
        img_dat = np.stack((img_c1, img_c2, img_c3), 3)
        return img_dat

if __name__ == '__main__':
    #load data
    import matplotlib.pyplot as plt
    import experiments.Segmentation_Bin_OARs as expConfig
    import systemsetup
    path = "/media/root/01456da2-1f1d-4b67-810e-b9cd3341133d/NPC_MICCAI_2015_original_data/HaN_2015_crop/test_all_headonly_noresample_new"
    trainset = OARDataset(path, expConfig, mode="train", mask_num = 9, hasMasks=True, AutoChangeDim=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=expConfig.BATCH_SIZE, shuffle=True, pin_memory=False, num_workers=expConfig.DATASET_WORKERS)

    for ii, sample in enumerate(trainloader):
        img = sample[0]
        gt = sample[2]

        img_i = img[0].numpy()
        img_local = img[1].numpy()
        gt = gt[0].numpy()

        print(np.max(gt))

        print(sample[1])

        for mi in range(9):
            if np.sum(gt[0, mi, :, :, :]) == 0:
                continue
            count_i = 0
            print(mi)
            for i in range(img_i.shape[4]):
                if np.sum(gt[0, mi, :, :, i]) > 0:
                    print(mi, '-', i)
                    plt.figure()
                    plt.title('display')
                    plt.subplot(131)
                    plt.imshow(img_i[0, 0, :, :, i])
                    plt.subplot(132)
                    plt.imshow(img_local[0, mi, :, :, i])
                    plt.subplot(133)
                    plt.imshow(gt[0, mi, :, :, i])
                    plt.show(block=True)

                    count_i += 1
                    if count_i >= 5:
                        break
