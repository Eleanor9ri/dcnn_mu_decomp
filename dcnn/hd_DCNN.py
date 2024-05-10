# build model
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
# from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import Callback, TensorBoard, EarlyStopping, ModelCheckpoint, LambdaCallback

# customized callback function to calculate averaged f1_score and accuracy across all outputs
class AccuracyCallback(Callback):
    def __init__(self, metric_name = 'accuracy'):
        super().__init__()
        self.metric_name = metric_name
        self.val_metric = []
        self.metric = []
        self.val_metric_mean = 0
        self.metric_mean = 0
        self.best_metric = 0
        
    def on_epoch_end(self, epoch, logs=None):
#         print('Accuracycallback')
        # extract values from logs
        self.val_metric = []
        self.metric = []
        for log_name, log_value in logs.items():
            if log_name.find(self.metric_name) != -1:
                if log_name.find('val') != -1:
                    self.val_metric.append(log_value)
                else:
                    self.metric.append(log_value)

        self.val_metric_mean = np.mean(self.val_metric)
        self.metric_mean = np.mean(self.metric)
        logs['val_{}'.format(self.metric_name)] = np.mean(self.val_metric)   # replace it with your metrics
        logs['{}'.format(self.metric_name)] = np.mean(self.metric)   # replace it with your metrics
  
# f1 score
def f1_m(y_true, y_pred):
#     precision = precision_m(y_true, y_pred)
#     recall = recall_m(y_true, y_pred)
    y_pred_binary = tf.where(y_pred>=0.5, 1., 0.)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred_binary, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred_binary, 0, 1)))
    
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    return 2*((precision*recall)/(precision + recall + K.epsilon()))

### keras.sequential ###
# create cnn with sequential model
def get_cnn1d_model(shape_in, shape_out, nn_nodes = [128, 128, 128, 64, 256]):
    '''Create a keras model.'''
    # shape_in = (timesteps, features)
    model = Sequential()
    gg_nn_nodes = nn_nodes
    print(gg_nn_nodes)
    
    #add model layers, number of filter, kernel_size
    model.add(Conv1D(filters=gg_nn_nodes[0], kernel_size= 3, activation= 'relu', input_shape=shape_in))
    model.add(Conv1D(filters=gg_nn_nodes[1], kernel_size= 3, activation= 'relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    model.add(Conv1D(filters=gg_nn_nodes[2], kernel_size=3, activation= 'relu'))
    model.add(Conv1D(filters=gg_nn_nodes[3], kernel_size=3, activation= 'relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(gg_nn_nodes[4], activation= 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(shape_out, activation= 'sigmoid'))

    return model

### keras.Functional_API ###
# create cnn with API interface; create models with given input shape and output shape
def get_cnn1d_api(shape_in, shape_out, nn_nodes = [128, 128, 128, 64, 256]):
    '''Create a keras model with functional API'''
    # create convolutional neural network model
    # shape_in = (timesteps, features)
    print(nn_nodes)
    
    # create shared layers
    visible = Input(shape = shape_in, name='EMG')
    cnn = Conv1D(filters=nn_nodes[0], kernel_size=3, activation='relu')(visible)
    cnn = Conv1D(filters=nn_nodes[1], kernel_size=3, activation='relu')(cnn)
    cnn = MaxPooling1D(pool_size=2)(cnn)
    cnn = Dropout(0.5)(cnn)
    
    # create seperate layers for each motor unit
    outputs = []
    for k in range(1, shape_out+1):
        cnn_2 = Conv1D(filters=nn_nodes[2], kernel_size=3, activation='relu')(cnn)
        cnn_2 = Conv1D(filters=nn_nodes[3], kernel_size=3, activation='relu')(cnn_2)
        cnn_2 = MaxPooling1D(pool_size=2)(cnn_2)
        cnn_2 = Dropout(0.5)(cnn_2)

        cnn_2 = Flatten()(cnn_2)
        s2 = Dense(nn_nodes[4], activation='relu')(cnn_2)
        s2 = Dropout(0.5)(s2)
        output = Dense(1, activation='sigmoid', name='output_{}'.format(k))(s2)
        outputs.append(output)

    # construct metrics and loss configuration
    metrics = {'output_1':['accuracy', f1_m]}
    loss = {'output_1':'binary_crossentropy'}   # ['accuracy', <function f1_m at 0x0000024579E41318>]
    for k in range(2, shape_out+1):
        key = 'output_{}'.format(k)
        metrics[key] = ['accuracy', f1_m]   # {'output_1': ['accuracy', <function f1_m at 0x0000024579E41318>], 'output_2': ['accuracy', <function f1_m at 0x0000024579E41318>], 'output_3': ['accuracy', <function f1_m at 0x0000024579E41318>], 'output_4': ['accuracy', <function f1_m at 0x0000024579E41318>]}
        loss[key]= 'binary_crossentropy'    # {'output_1': 'binary_crossentropy', 'output_2': 'binary_crossentropy', 'output_3': 'binary_crossentropy', 'output_4': 'binary_crossentropy'}

    # tie together
    model = Model(inputs=visible, outputs=outputs)
    return model, loss, metrics

### build model ###
def build_model(WS = 120, n_output = 1, nn_nodes = [128, 128, 128, 64, 256]):
    # mIndex is not in use
    n_input = WS    # WS: window_size
    n_features = 64 # set default number of EMG channels(默认EMG通道数)

    ## 选择创建模式
    # 一维输出
    if n_output == 1:
        print('Sequential model')
        model_cnn = get_cnn1d_model((n_input, n_features), n_output, nn_nodes)
        loss_cnn = 'binary_crossentropy'
        metrics_cnn = [
            'accuracy',
            'mse',
             f1_m,
            ]
    # 多维输出
    else:
        print('API model')
        model_cnn, loss_cnn, metrics_cnn = get_cnn1d_api((n_input, n_features), n_output, nn_nodes)

    model = model_cnn
    model.compile(optimizer = 'rmsprop', #sgd', 'adagrad', 'rmsprop', 'adam'
                    loss = loss_cnn,  # mean_squared_error
                    metrics = metrics_cnn) #['accuracy', 'mse'])
    return model_cnn

### train model ###
def train_model(model, x_data, y_data, prefix, epochs = 100):
    tname = int(time.time())    # 时间戳
    batch_size = 64
    
    # create tersorboard
    log_name = "hdEMG_{}_{}".format(prefix, tname)  # 'hdEMG_cnn-5_50_GM-SG0-ST20-WS120-MU[0, 1, 2, 3]_1715005200'
    tensorboard = TensorBoard(log_dir = ".\\logs\\{}".format(log_name))
    # check tenorboard by running 
    # tensorboard --logdir C:\Users\Yue\Documents\mudecomp\logs\ # in the anaconda prompt command line
    # tensorboard --logdir C:\Users\ywen.SMPP\Documents\mudecomp\logs\
    
    # early stop when loss improvement is small
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=50)
    
    # 检查并创建模型保存目录
    model_dir = '.\\model\\'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # 构建模型文件路径
    model_name = os.path.join(model_dir, 'best_model_{}_{}_l.h5'.format(prefix, tname))

    # save the best model when loss is minimum and f1_score is highest
    # mc = ModelCheckpoint('best_model_{}_{}_l.h5'.format(prefix, tname), monitor='loss', mode='min', verbose=1, save_best_only=True)
    # mc_vl = ModelCheckpoint( 'best_model_{}_{}_vl.h5'.format(prefix, tname), monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    # mc_f = ModelCheckpoint('best_model_{}_{}_f.h5'.format(prefix, tname), monitor='f1_m', mode='max', verbose=1, save_best_only=True)
    # mc_vf = ModelCheckpoint('best_model_{}_{}_vf.h5'.format(prefix, tname), monitor='val_f1_m', mode='max', verbose=1, save_best_only=True)
    mc = ModelCheckpoint(model_name, monitor='loss', mode='min', verbose=1, save_best_only=True)
    mc_vl = ModelCheckpoint(os.path.join(model_dir, 'best_model_{}_{}_vl.h5'.format(prefix, tname)), monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    mc_f = ModelCheckpoint(os.path.join(model_dir, 'best_model_{}_{}_f.h5'.format(prefix, tname)), monitor='f1_m', mode='max', verbose=1, save_best_only=True)
    mc_vf = ModelCheckpoint(os.path.join(model_dir, 'best_model_{}_{}_vf.h5'.format(prefix, tname)), monitor='val_f1_m', mode='max', verbose=1, save_best_only=True)

    # create customized callbacks
    # 记录每个epoch的准确率和f1
    accuracy_callback = AccuracyCallback('accuracy')
    f1_callback = AccuracyCallback('f1_m')
    
    # train model
    history = model.fit(x_data, 
                        y_data,
                        validation_split = 0.2,
                        batch_size = batch_size,
                        epochs = epochs,
                        verbose = 1,
                        callbacks = [es, mc, mc_vl, accuracy_callback, f1_callback, tensorboard, mc_f, mc_vf])

    # return best model for further evaluation
    

    model = load_model(model_name, custom_objects={"f1_m": f1_m})   # model_name: file dir
    return model, tname

### validate modeal ###
# validate model with given data sets
def model_validate(model, x_data, y_data, prefix):
    # sequential data
    y_pred = evaluate(model, x_data, y_data)
    savedata(y_data, y_pred, "{}".format(prefix))
    
# evaluate model prediction
def evaluate(model, x_val, y_val, showFlag = 1):    # flag = 1
    # 
    print('\n# Generate predictions')
    y_pred = model.predict(x_val)
    y_pred = np.asarray(y_pred)
    if len(y_pred.shape) == 3:
        y_pred = np.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1]))
    
    if showFlag:
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(y_val)
        ax1.set_title('real_value')
        ax2.plot(y_pred)
        ax2.set_title('predict_value')
        plt.show()
    return y_pred

# save prediction and acutal values to csv file
def savedata(y_val, y_pred, fname):
    
    # convert to array if y_val type is list
    if type(y_val) is list:
        y_val = np.array(y_val)
    if type(y_pred) is list:
        y_pred = np.array(y_pred)
    
    # reshape
    if len(y_pred.shape) == 3:
        y_pred = np.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1]))
    if len(y_val.shape) == 3:
        y_val = np.reshape(y_val, (y_val.shape[0], y_val.shape[1]))

    # rotate
    if len(y_val.shape) == 2 and y_val.shape[0] < y_val.shape[1]:
        y_val = np.transpose(y_val)
    elif len(y_val.shape) == 1:
        y_val = np.reshape(y_val, (y_val.shape[0], 1))

    if y_pred.shape[0] < y_pred.shape[1]:
        y_val = np.transpose(y_val)

    # save data
    if  y_val.shape[0] > y_val.shape[1] and y_val.shape[0] == y_pred.shape[0]:
        data = np.column_stack((y_val, y_pred))
#         data = np.transpose(data)
    else:
        data = np.vstack((y_val, y_pred))
    
    if data.shape[0] < data.shape[1]:
        data = np.transpose(data)
    data.shape
    pd.DataFrame(data).to_csv("output-{}.csv".format(fname))

### display model ###
def load_model_custom(model_name):
    model = load_model(model_name, custom_objects={"f1_m": f1_m})
    return model

