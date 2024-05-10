import numpy as np
import scipy.io as sio
from os import path
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def load_data_mat (TR, SG, ST, MU, WS, TF = 0, MutiSeg = 0) :
    seg = [1, 2, 3]
    # load train data: SG = 1, 2
    # load test data:  SG = 3 
    segment = seg[SG]   # 1
    # mat file name
    prefix = "{}-SG{}-WS{}-ST{}".format(TR, segment, WS, ST)    # '5_50_GM-SG1-WS120-ST20'  SG1
    matfile = "{}.mat".format(prefix)                           # '5_50_GM-SG1-WS120-ST20.mat'

    # 如果不在当前文件夹下
    if not path.exists(matfile) :
        pathstr = 'D:\\paper\\emg_data\\14179583\\'   # data folder
        matfile = "{}{}".format(pathstr, matfile)
    
    vnames = ['EMGs', 'Spikes']
    # load mat file
    data = sio.loadmat(matfile, variable_names=vnames)
    x_data = data['EMGs']   # shape:(2613, 120, 64)
    spikes = data['Spikes'] # shape:(2613, 10)

    # load second segment if MutiSeg is 1
    if MutiSeg: # 1
        seg2 = [2, 3, 1]
        segment = seg2[SG]  # 2
        prefix = "{}-SG{}-WS{}-ST{}".format(TR, segment, WS, ST)    # '5_50_GM-SG2-WS120-ST20'  SG2
        matfile = "{}.mat".format(prefix);                          # '5_50_GM-SG2-WS120-ST20.mat'
        if not path.exists(matfile):
            pathstr = 'D:\\paper\\emg_data\\14179583\\'
            matfile = "{}{}".format(pathstr, matfile)
    #     print(matfile)
        data_2 = sio.loadmat(matfile, variable_names=vnames)
        x_data_2 = data_2['EMGs']   # shape:(2547, 120, 64)
        spikes_2 = data_2['Spikes'] # shape:(2547, 10)
        x_data = np.concatenate((x_data, x_data_2))     # shape:(5160, 120, 64)
        spikes = np.concatenate((spikes, spikes_2))     # shape:(5160, 10)

#     x_data.shape
    # exactract spikes for given motor units
    if type(MU) is list:    # MU:[0, 1, 2, 3] true
        y_data = []
        for c in MU:    # c:0,1,2,3
            if c < spikes.shape[1]: # spikes.shape[1]:10
                y_data.append(spikes[:, c])
            else:
                y_data.append(spikes[:, -1]*0)
    else:
        y_data = []
        y_data.append(spikes[:, MU])

    ## shuffle the data based on TF flag
    y_data = np.array(y_data)   # shape:(4, 5160)
    y_data = y_data.T           # shape:(5160, 4)
    if TF == 1:     # 1
        # 乱序(转置的目的：shuffle后保持x和y的对应关系不变)
        x_data, y_data = shuffle(x_data, y_data)
    elif TF > 0: 
        x_data, _, y_data, _= train_test_split(x_data, y_data, test_size = 1.0-TF)
    else:
        print('no shuffle')
    y_data = y_data.T
    y_data = list(y_data)   # len():4

    return x_data, y_data

