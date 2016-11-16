import scipy.io
import numpy as np
mat = scipy.io.loadmat('vidf1_33_000_count_roi_mainwalkway.mat')

def get_target(mat = mat):
    count = mat['count'][0]
    count_r = np.array(count[0][0])
    count_l = np.array(count[1][0])
    res = count_l+count_r
    return res

if __name__== "__main__":
    get_target()