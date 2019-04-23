import os

if __name__ == "__main__":
    path = '/media/root/01456da2-1f1d-4b67-810e-b9cd3341133d/AI challenge/2018_seg/Edema_trainingset/'
    pathH5 = '/media/root/01456da2-1f1d-4b67-810e-b9cd3341133d/AI challenge/2018_seg/Edema_trainingset/train_h5/'
    path_train = path+'original_images/'
    path_label = path+'label_images/'
    list_data = os.listdir(path_train)
    data_num = len(list_data)
    for a in range(data_num):
        data_name = list_data[a]
        data_onepath = path_train+data_name+'/'
        label_onepath = path_label+data_name[:-3]+'_labelMark/'
        for num in range(1,129):
            data_fullpath = data_onepath+str(num)+'.bmp'
            label_fullpath = label_onepath + str(num) + '.bmp'
            data_1 = open(data_fullpath, 'rb')
            label_1 = open(label_fullpath, 'rb')


