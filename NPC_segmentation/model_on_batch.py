## Bin Huang, made in 2018.8.31

import random
from keras.callbacks import *
import read_data
import time
import tensorboard_log
import copy

def seg_model_train(DataH5Path,ModelPath,GraphPath,model,*args,input_num=1,output_num=1,epoch=1,
                batchsize=1,save_model_mode='iteration',save_per=1000,use_tensorboard=False,tensorboard_graph_save=200):
    start_time = time.time()
    H5_list = os.listdir(DataH5Path)
    H5_num = len(H5_list)
    random.shuffle(H5_list)
    num = 0
    # read_num = len(args)
    data_input = []
    label_input = []
    for nn in range(input_num):
        name_num = nn+1
        exec ("data%s = []"%name_num)
    for nn in range(output_num):
        name_num = nn+1
        exec ("label%s = []"%name_num)

    if use_tensorboard:
        ## tensorboard setup
        writer = tensorboard_log.setup_tensorboard(GraphPath)

    bs_count = 0
    real_iteration = 0
    max_iteration = epoch*(H5_num//batchsize)
    if save_model_mode=='iteration':
        num_save = save_per
    elif save_model_mode=='epoch':
        num_save = H5_num
    else:
        print("save_model_mode only is 'iteration' or 'epoch'.")
        return 0

    print('Training dataset num: ',H5_num)
    print('Total epoch: ',epoch)
    print('Read data: ',args)

    while True:
        H5_name = H5_list[num]
        Data_fullpath = DataH5Path+H5_name
        data_list = read_data.read_data_h5(Data_fullpath,*args)

        for nn in range(input_num):
            name_num = nn + 1
            exec("data%s.append(data_list[%d])" %(name_num,nn))
        for nn in range(output_num):
            name_num = nn + 1
            exec("label%s.append(data_list[%d])" %(name_num,nn+input_num))
        bs_count+=1
        num+=1

        ## shuffle the data list per epoch
        if num>=H5_num:
            num = 0
            random.shuffle(H5_list)
        if bs_count<batchsize:
            continue

        for nn in range(input_num):
            name_num = nn + 1
            exec("data%s = np.array(data%s)" %(name_num,name_num))
            exec("data_input.append(data%s)" %name_num)
        for nn in range(output_num):
            name_num = nn + 1
            exec("label%s = np.array(label%s)" %(name_num,name_num))
            exec("label_input.append(label%s)" %name_num)

        real_iteration += 1

        result_threeD = model.train_on_batch(data_input,label_input)
        print(H5_name,"iteration ", real_iteration, " Result: ", result_threeD)

        if use_tensorboard:
            ## tensorboard data save
            tensorboard_log.tensorboard_log_losses(result_threeD[0], 'loss', writer, real_iteration+1)

        # result_threeD = model.test_on_batch(data_input,label_input)
        # print(H5_name,"iteration ", real_iteration, " Result_after: ", result_threeD)
        # print('\n')

        if use_tensorboard:
            if (real_iteration)%tensorboard_graph_save == 0:
                ## tensorboard graph save
                tensorboard_log.tensorboard_log_image(np.array(data_input[0]), 'result', writer, real_iteration)

        ## initialization
        for nn in range(input_num):
            name_num = nn + 1
            exec("data%s = []" % name_num)
        for nn in range(output_num):
            name_num = nn + 1
            exec("label%s = []" % name_num)
        data_input = []
        label_input = []
        bs_count = 0



        ## saving weights
        if real_iteration%num_save==0:
            str_num = str(real_iteration)
            print('Saving weights!')
            model.save(ModelPath + 'weights_' + str_num + '.h5')

        ## Stop training
        if real_iteration >= max_iteration:
            end_time = time.time()
            print("Total time: ",end_time-start_time)
            break

def seg_model_multi_gpu_train(DataH5Path,ModelPath,GraphPath,model_multi_gpu,model,*args,input_num=1,output_num=1,epoch=1,
                batchsize=1,save_model_mode='iteration',save_per=1000,use_tensorboard=False,tensorboard_graph_save=200):
    start_time = time.time()
    H5_list = os.listdir(DataH5Path)
    H5_num = len(H5_list)
    random.shuffle(H5_list)
    num = 0
    # read_num = len(args)
    data_input = []
    label_input = []
    for nn in range(input_num):
        name_num = nn+1
        exec ("data%s = []"%name_num)
    for nn in range(output_num):
        name_num = nn+1
        exec ("label%s = []"%name_num)

    if use_tensorboard:
        ## tensorboard setup
        writer = tensorboard_log.setup_tensorboard(GraphPath)

    bs_count = 0
    real_iteration = 0
    max_iteration = epoch*(H5_num//batchsize)
    if save_model_mode=='iteration':
        num_save = save_per
    elif save_model_mode=='epoch':
        num_save = H5_num
    else:
        print("save_model_mode only is 'iteration' or 'epoch'.")
        return 0

    print('Training dataset num: ',H5_num)
    print('Total epoch: ',epoch)
    print('Max iteration: ', max_iteration)
    print('Read data: ',args)

    while True:
        H5_name = H5_list[num]
        Data_fullpath = DataH5Path+H5_name
        data_list = read_data.read_data_h5(Data_fullpath,*args)

        for nn in range(input_num):
            name_num = nn + 1
            exec("data%s.append(data_list[%d])" %(name_num,nn))
        for nn in range(output_num):
            name_num = nn + 1
            exec("label%s.append(data_list[%d])" %(name_num,nn+input_num))
        bs_count+=1
        num+=1

        ## shuffle the data list per epoch
        if num>=H5_num:
            num = 0
            random.shuffle(H5_list)
        if bs_count<batchsize:
            continue

        for nn in range(input_num):
            name_num = nn + 1
            exec("data%s = np.array(data%s)" %(name_num,name_num))
            exec("data_input.append(data%s)" %name_num)
        for nn in range(output_num):
            name_num = nn + 1
            exec("label%s = np.array(label%s)" %(name_num,name_num))
            exec("label_input.append(label%s)" %name_num)

        real_iteration += 1

        result_threeD = model_multi_gpu.train_on_batch(data_input,label_input)
        print(H5_name,"iteration ", real_iteration, " Result: ", result_threeD)

        if use_tensorboard:
            ## tensorboard data save
            tensorboard_log.tensorboard_log_losses(result_threeD[0], 'loss', writer, real_iteration+1)

        # result_threeD = model.test_on_batch(data_input,label_input)
        # print(H5_name,"iteration ", real_iteration, " Result_after: ", result_threeD)
        # print('\n')

        if use_tensorboard:
            if (real_iteration)%tensorboard_graph_save == 0:
                ## tensorboard graph save
                tensorboard_log.tensorboard_log_image(np.array(data_input[0]), 'result', writer, real_iteration)

        ## initialization
        for nn in range(input_num):
            name_num = nn + 1
            exec("data%s = []" % name_num)
        for nn in range(output_num):
            name_num = nn + 1
            exec("label%s = []" % name_num)
        data_input = []
        label_input = []
        bs_count = 0



        ## saving weights
        if real_iteration%num_save==0:
            str_num = str(real_iteration)
            print('Saving weights!')
            model.save(ModelPath + 'weights_' + str_num + '.h5')

        ## Stop training
        if real_iteration >= max_iteration:
            end_time = time.time()
            print("Total time: ",end_time-start_time)
            break


def seg_model_test(DataH5Path, model, *args, input_num=1, output_num=1,seg_thresh=0.5):
    start_time = time.time()
    epsilon = 1e-7
    H5_list = os.listdir(DataH5Path)
    H5_num = len(H5_list)
    random.shuffle(H5_list)
    num = 0
    # read_num = len(args)
    data_input = []
    label_input = []
    for nn in range(input_num):
        name_num = nn + 1
        exec("data%s = []" % name_num)
    for nn in range(output_num):
        name_num = nn + 1
        exec("label%s = []" % name_num)

    bs_count = 0
    real_iteration = 0
    max_iteration = H5_num

    print('Test dataset num: ', H5_num)
    print('Read data: ', args)
    tp_all = 0
    fp_all = 0
    tn_all = 0
    fn_all = 0
    while True:
        H5_name = H5_list[num]
        Data_fullpath = DataH5Path + H5_name
        data_list = read_data.read_data_h5(Data_fullpath, *args)

        for nn in range(input_num):
            name_num = nn + 1
            exec("data%s.append(data_list[%d])" % (name_num, nn))
        for nn in range(output_num):
            name_num = nn + 1
            exec("label%s.append(data_list[%d])" % (name_num, nn + input_num))
        bs_count += 1
        num += 1

        for nn in range(input_num):
            name_num = nn + 1
            exec("data%s = np.array(data%s)" % (name_num, name_num))
            exec("data_input.append(data%s)" % name_num)
        for nn in range(output_num):
            name_num = nn + 1
            exec("label%s = np.array(label%s)" % (name_num, name_num))
            exec("label_input.append(label%s)" % name_num)

        real_iteration += 1

        result_threeD = model.test_on_batch(data_input, label_input)
        print(H5_name, "Number ", real_iteration, " test_index: ", result_threeD)
        result_image = model.predict_on_batch(data_input)
        result_image_bw = copy.deepcopy(result_image)
        result_image_bw[result_image_bw>=seg_thresh] = 1
        result_image_bw[result_image_bw<seg_thresh] = 0
        result_shape = np.shape(result_image_bw)
        pred_sum = np.sum(result_image_bw)
        true_sum = np.sum(label_input)
        all_sum = np.sum(np.ones_like(result_image_bw))

        tp = np.sum(result_image_bw*label_input)
        fp = pred_sum-tp
        fn = true_sum-tp
        tn = all_sum-tp-fp-fn

        tp_all+=tp
        fp_all+=fp
        fn_all+=fn
        tn_all+=tn


        if real_iteration==1:
            result_total = result_threeD
        else:
            result_total+=result_threeD


        for nn in range(input_num):
            name_num = nn + 1
            exec("data%s = []" % name_num)
        for nn in range(output_num):
            name_num = nn + 1
            exec("label%s = []" % name_num)
        data_input = []
        label_input = []

        if real_iteration >= max_iteration:
            # result_total = result_total/H5_num
            recall = tp_all/(tp_all+fn_all+epsilon)
            precision = tp_all/(tp_all+fp_all+epsilon)
            dsc = (2*recall*precision)/(recall+precision+epsilon)
            print('recall:',recall,' precision:',precision,' dsc:',dsc)
            end_time = time.time()
            print("Total time: ", end_time - start_time)
            break

##test code
# if __name__ == "__main__":
#     train_PathH5 = '/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation/data_h5/train_test_fit_gen/'
#     model_train(train_PathH5,1,'t2','gt')
