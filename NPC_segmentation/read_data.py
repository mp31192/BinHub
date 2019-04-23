## Bin Huang, made in 2018.8.31

import h5py

def read_data_h5(path,*args):
    ## 'path': the fullpath of h5, etc: /root/*.h5
    read_num = len(args)
    read_list = []

    h5_file = h5py.File(path,'r')
    for nn in range(read_num):
        read_data = h5_file[args[nn]][:]
        read_list.append(read_data)
    h5_file.close()

    return read_list

#
# if __name__ == "__main__":
#     data_list = read_data_h5('/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation/data_h5/train_test_fit_gen/27_01.h5','t2','gt')