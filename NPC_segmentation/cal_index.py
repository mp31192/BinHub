## Bin Huang, made in 2018.8.31

epsilon = 1e-7
def cal_seg_index(result,test_num):
    tp = result[1]*test_num
    tn = result[2]*test_num
    fp = result[3]*test_num
    fn = result[4]*test_num
    recall,precision,dsc = cal_rpd(tp,tn,fp,fn)
    return recall,precision,dsc

def cal_rpd(tp,tn,fp,fn):
    recall = tp/(tp+fn+epsilon)
    precision = tp/(tp+fp+epsilon)
    dsc = (2*recall*precision)/(recall+precision+epsilon)
    return recall,precision,dsc