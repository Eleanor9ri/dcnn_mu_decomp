import pandas as pd
import os
from os import path
import glob
import time
import numpy as np
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog

from hd_LOAD import load_data_mat
from hd_DCNN import build_model, train_model, model_validate, load_model_custom

# dir
print(os.getcwd())

# initialize parameters
trial = '5_50_GM'
train_seg = 0
test_seg = 2
step_size = 20
window_size = 120
mu = [0, 1, 2, 3]
mucnt = len(mu)

#################### build model ######################
model = build_model(WS = window_size, n_output = mucnt) # n_output = 4
model.summary() # print summary

#################### train model ######################
# load train data
x_train, y_train = load_data_mat(TR = trial, SG = train_seg, ST = step_size, WS = window_size, MU = mu, TF = 1, MutiSeg = 1)
# x_train  # shape:(5160, 120, 64)
# y_train  # len():4 -- 4*(5160,)
prefix = "cnn-{}-SG{}-ST{}-WS{}-MU{}".format(trial, train_seg, step_size, window_size, mu)  # cnn-5_50_GM-SG0-ST20-WS120-MU[0, 1, 2, 3]
print(prefix)
model, tname = train_model(model, x_train, y_train, prefix, epochs = 10)   # epochs = 100

#################### test model ######################
# load testing data
x_test, y_test = load_data_mat(TR = trial, SG = test_seg, ST = step_size, WS = window_size, MU = mu)
# validate model and save the output as csv file
prefix4test = "{}-TSG{}".format(prefix, test_seg)
model_validate(model, x_test, y_test, prefix4test)

####################### real-time HD-EMG decomposition ########################
# gui tkinter
def gui_folder(dir = None):
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(initialdir=dir, title="Select Directory")
    return folder


def gui_fname(dir=None):
    """Select a file via a dialog and return the file name."""
    if dir is None: dir ='./'
    fname = filedialog.askopenfile(initialdir=dir, title="Select data file...", 
                filetypes=(("All files", "*.*"), ("SM Files", "*.sm")))
    return fname

######################## motor unit decomposer ####################
class MUdecomposer(object):
    def __init__(self, model_file = None):
        if model_file == None:
            model_file = gui_fname()
            
        self.model_file = model_file
        # load model from h5 file
        self.model = load_model_custom(model_file)

    def predict_MUs(self, hdEMG):
        # predict and generate output
        self.preds = self.model.predict(hdEMG)
        self.preds_binary = tf.where(np.array(self.preds)>=0.5, 1., 0.)
        return self.preds_binary

######################### hdEMG generator to provide HDEMG signals ####################
class hdEMG(object):
    def __init__(self, fileName = None):
        if fileName == None:
            matfile = gui_fname('D:\\paper\\emg_data\\14179583\\')
        else:
            matfile = fileName
            if not path.exists(fileName):
                pathstr = 'D:\\paper\\emg_data\\14179583\\'
                matfile = "{}{}".format(pathstr, fileName)
                
        # load EMG data from mat file
        #     print(matfile)
        self.matfile = matfile
        data = sio.loadmat(matfile)
        self.EMGs = data['EMGs']
        self.index = 0
        self.frameCnt = self.EMGs.shape[0]

    def reset_frame(self):
        self.index = 0
    
    # returns hdEMG frames
    def get_frame(self, index = None):
        if index == None:
            EMG = self.EMGs[self.index:self.index+1, :, :]
            self.index = self.index + 1
        else:
            self.index = index
            EMG = self.EMGs[self.index:self.index+1, :, :]
            self.index = self.index + 1
#         print(EMG.shape)
        return EMG
    
####################### real-time prediction to get processing time ########################
def processing_Time(modelFile = None, matFile = None, frames = 500):
    if modelFile == None:
        modelFile = gui_fname()
    if matFile == None:
        model_file = modelFile.split('/')[-1]
        model_file = model_file.split('\\')[-1]
        modelItem = model_file.split('-')
        matFile = "{}-SG{}-{}-{}.mat".format(modelItem[1], int(modelItem[2][-1])+1, modelItem[4], modelItem[3])
    print("{} \n {}".format(model_file, matFile))
    
    # load model and mat file
    mude = MUdecomposer(modelFile)
    EMGstream = hdEMG(matFile)
    EMG = EMGstream.get_frame()
#     maxK = EMGstream.frameCnt
    mude.predict_MUs(EMG)
    
    if frames == 0:
        frames = EMGstream.frameCnt
    
    # predict with each frame
    tHist = []
    spike = []
    EMGstream.reset_frame()
    for k  in range(1, frames):
        EMG = EMGstream.get_frame()
        start_time = time.time()
        s = mude.predict_MUs(EMG)
        tHist.append(time.time() - start_time)
        spike.append(s)
    print("--- %s seconds ---" % np.mean(tHist))
    plt.figure()
    plt.plot(tHist, 'r', label='time')
    plt.show()
    return tHist, mude, EMG

# get model file
modelFilter = ".\\model\\best_model*SG{}*ST{}*WS{}*_f.h5".format(train_seg, step_size, window_size)
print(modelFilter)
modelFiles = glob.glob(modelFilter)
modelFile = modelFiles[0]
print(modelFile)
for modelFile in modelFiles[0:]:
    pItem = modelFile.split('\\')                               # ['.', 'model', 'best_model_cnn-5_50_GM-SG0-ST20-WS120-MU[0, 1, 2, 3]_1715053562_f.h5']
    pPath = "\\".join(pItem[:-1])                               # '.\\model'
    Tstamp = pItem[-1].split('_')[-2]                           # '1715053562'
    matFile = "{}\\Processtime_{}.mat".format(pPath, Tstamp)    # '.\\model\\Processtime_1715053562.mat'
    tHist, _, _ = processing_Time(modelFile, frames = 0)
    sio.savemat(matFile, {"pTime":tHist})