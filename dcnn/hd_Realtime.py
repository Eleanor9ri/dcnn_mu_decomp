import glob
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io as sio
import time
import numpy as np
from os import path
import glob
import gc

from hd_DCNN import load_model_custom

# gui tkinter
import tkinter as tk
from tkinter import filedialog
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
#     plt.figure()
#     plt.plot(tHist, 'r', label='time')
#     plt.show
    return tHist, mude, EMG

# get model file
modelFilter = ".\\model\\best_model*SG{}*ST{}*WS{}*_f.h5".format(train_seg, step_size, window_size)
print(modelFilter)
modelFiles = glob.glob(modelFilter)
modelFile = modelFiles[0]
print(modelFile)
for modelFile in modelFiles[0:]:
    pItem = modelFile.split('\\')
    pPath = "\\".join(pItem[:-1])
    Tstamp = pItem[-1].split('_')[-2]
    matFile = "{}\\Processtime_{}.mat".format(pPath, Tstamp)
    tHist, _, _ = processing_Time(modelFile, frames = 0)
    sio.savemat(matFile, {"pTime":tHist})