import os
import numpy as np
from pyparsing import Char
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
import wandb
from wandb.keras import WandbCallback
import argparse

def make_parser():
    parser = argparse.ArgumentParser("Counting algorithm")
    #gpu_num
    parser.add_argument(
        "-g", "--gpu_num",
        type=str,
        default='0',
        help="Input gpu_number"
    )
    #network_model
    parser.add_argument(
        "-m", "--network_model",
        type = Char,
        default= 'F',
        help="Input network_model\nFC_network = F\nConv_network = C\nAlex_network = A\nLe_network = L\nVgg-19_network = V\nResnet18_network = R"
    )
    #learning_rate
    parser.add_argument(
        "-lr", "--learning_rate",
        type = float,
        default=0.98,
        help="Input learning_rate"
    )
    #epoch
    parser.add_argument(
        "-e", "--epoch",
        type = int,
        default=100,
        help="Input epoch"
    )
    #start_lr
    parser.add_argument(
        "-s", "--start_lr",
        type = float,
        default=1e-4,
        help="Input start_lr"
    )
    #epoch_drop
    parser.add_argument(
        "-d", "--epoch_drop",
        type = int,
        default = 1,
        help="Input epoch_drop"
    )
    #drop_rate
    parser.add_argument(
        "-dr", "--drop_rate",
        type = float,
        default = 0.9,
        help="Input drop_rate"
    )
    return parser

if __name__ == "__main__" :
    args = make_parser().parse_args()
    gpu_num = args.gpu_num
    learning_rate = args.learning_rate
    epoch = args.epoch
    model =args.network_model
    strat_lr = args.start_lr
    epoch_drop = args.epoch_drop
    drop_rate = args.drop_rate

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num

realCIR = np.load('/home/real.npy')
imagCIR = np.load('/home/imag.npy')
scaler_real = MinMaxScaler()
scaler_imag = MinMaxScaler()
scaler_real.fit(realCIR)
scaler_imag.fit(imagCIR)
realCIR = scaler_real.transform(realCIR)
imagCIR = scaler_imag.transform(imagCIR)
magCIR = np.load('/home/dataset.npy')
GT = np.load('/home/label.npy')
GT = GT.astype(int)
tan = np.arctan2(imagCIR, realCIR)
# diff = tan[:,:63] - tan[:,1:]
# diffrence = np.zeros((magCIR.shape[0],magCIR.shape[1]))
# array = np.stack([magCIR,diffrence],axis=-1)
array = np.stack([realCIR,imagCIR],axis=-1)

# print(realCIR.shape, imagCIR.shape, magCIR.shape, GT.shape)

X_train, X_test, y_train, y_test = train_test_split(array, GT, test_size=0.3, shuffle=True, stratify=GT, random_state=48) 
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=True, stratify=y_test, random_state=48)

# np.save('X_train.npy',X_train)
# np.save('X_val.npy',X_val)
# np.save('X_test.npy',X_test)
# np.save('y_train.npy',y_train)
# np.save('y_val.npy',y_val)
# np.save('y_test.npy',y_test)
 
# X_train = array[:27000,:]
# y_train = GT[:27000]
# X_val = array[27000:30000,:]
# y_val = GT[27000:30000]
# X_test = array[30000:,:]
# y_test = GT[30000:]

def step_decay(epoch):
    start = strat_lr
    drop = drop_rate
    epochs_drop = epoch_drop
    lr = start * (learning_rate ** np.floor((epoch)/epochs_drop))
    return lr

lr_scheduler = LearningRateScheduler(step_decay, verbose=1) 
# lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, cooldown=5, mode=min)
earlystopping = EarlyStopping(monitor='val_accuracy', patience=10, mode=max, verbose=1, restore_best_weights=True)

result = []
# parameter setting
if model == 'F':
    model_name = "FCNet 1layer "
    p = [
        [10],
        [20],
        [30],
        [40],
        [50],
        [60],
        [70],
        [80],
        [90],
        [100],
        [110],
        [120],
        [130],
        [140],
        [150],
        [160],
        [170],
        [180],
        [190],
        [200],
        [250],
        [300],
        [350],
        [400],
        [500],
        [600],
        [700],
        [800],
        [900],
        [1000],
        [1100],
        [1200],
        [1300],
        [1400],
        [1500],
        [1600]]
elif model == '2':
    model_name = "FCNet 2layer "
    p = [
        [64,64],
        [64,128],
        [128,64],
        [128,128],
        [64,16],
        [16,64],
        [16,16],
        [16,128],
        [128,16],
        [64,256],
        [256,64],
        [256,256],
        [16,256],
        [256,16],
        [128,256],
        [256,16],
        [64,400],
        [400,64],
        [400,400],
        [16,400],
        [400,16],
        [128,400],
        [400,128],
        [256,400],
        [400,256],
        [64,800],
        [800,64],
        [800,800],
        [16,800],
        [800,16],
        [128,800],
        [800,128],
        [256,800],
        [800,256],
        [400,800],
        [800,400]]
elif model == '3':
    model_name = "FCNet 3layer "
    p = [
        [64,64,64],
        [64,64,128],
        [64,128,64],
        [128,64,64],
        [64,128,128],
        [128,128,64],
        [128,64,128],
        [128,128,128],
        [64,64,256],
        [64,256,64],
        [256,64,64],
        [64,256,256],
        [256,256,64],
        [256,64,256],
        [256,256,256],
        [64,64,16],
        [64,16,64],
        [16,64,64],
        [64,16,16],
        [16,16,64],
        [16,64,16],
        [16,16,16],
        [256,128,64],
        [64,128,256],
        [128,256,256],
        [256,256,128],
        [256,128,256],
        [16,64,128],
        [128,64,16],
        [400,400,400],
        [400,400,800],
        [400,800,400],
        [800,400,400],
        [400,800,800],
        [800,800,400],
        [800,400,800],
        [800,800,800],
        [200,200,200],
        [200,200,400],
        [200,400,200],
        [400,200,200],
        [200,400,400],
        [400,400,200],
        [400,200,400],
        [200,400,800],
        [800,400,200],
        [800,800,200],
        [200,800,800],
        [800,200,800],
        [200,200,800],
        [800,200,200],
        [200,800,200]]
elif model == 'C1':
    model_name = "ConvNet 1layer "
    p = [
        [64,3],
        [128,3],
        [256,3],
        [400,3],
        [800,3],
        [64,5],
        [128,5],
        [256,5],
        [400,5],
        [800,5],
        [64,7],
        [128,7],
        [256,7],
        [400,7],
        [800,7],
        [64,9],
        [128,9],
        [256,9],
        [400,9],
        [800,9],
        [64,11],
        [128,11],
        [256,11],
        [400,11],
        [800,11],
        [64,13],
        [128,13],
        [256,13],
        [400,13],
        [800,13],
        [64,15],
        [128,15],
        [256,15],
        [400,15],
        [800,15]]
elif model == 'C2X':
    model_name = "ConvNet 2layer"
    p = [
        [64,3,64,3],
        [64,3,128,3],
        [64,3,256,3],
        [128,3,64,3],
        [256,3,64,3],
        [128,3,128,3],
        [128,3,256,3],
        [256,3,128,3],
        [256,3,256,3],
        [256,3,400,3],
        [400,3,256,3],
        [400,3,400,3],
        [400,3,800,3],
        [800,3,400,3],
        [800,3,800,3],
        [64,3,64,5],
        [64,3,128,5],
        [64,3,256,5],
        [128,3,64,5],
        [256,3,64,5],
        [128,3,128,5],
        [128,3,256,5],
        [256,3,128,5],
        [256,3,256,5],
        [256,3,400,5],
        [400,3,256,5],
        [400,3,400,5],
        [400,3,800,5],
        [800,3,400,5],
        [800,3,800,5],
        [64,5,64,3],
        [64,5,128,3],
        [64,5,256,3],
        [128,5,64,3],
        [256,5,64,3],
        [128,5,128,3],
        [128,5,256,3],
        [256,5,128,3],
        [256,5,256,3],
        [256,5,400,3],
        [400,5,256,3],
        [400,5,400,3],
        [400,5,800,3],
        [800,5,400,3],
        [800,5,800,3],
        [64,5,64,5],
        [64,5,128,5],
        [64,5,256,5],
        [128,5,64,5],
        [256,5,64,5],
        [128,5,128,5],
        [128,5,256,5],
        [256,5,128,5],
        [256,5,256,5],
        [256,5,400,5],
        [400,5,256,5],
        [400,5,400,5],
        [400,5,800,5],
        [800,5,400,5],
        [800,5,800,5]]
elif model == 'C2Y':
    model_name = "ConvNet 2layer Y"
    p = [
        [64,5,64,7],
        [64,5,128,7],
        [64,5,256,7],
        [128,5,64,7],
        [256,5,64,7],
        [128,5,128,7],
        [128,5,256,7],
        [256,5,128,7],
        [256,5,256,7],
        [256,5,400,7],
        [400,5,256,7],
        [400,5,400,7],
        [400,5,800,7],
        [800,5,400,7],
        [800,5,800,7],
        [64,7,64,5],
        [64,7,128,5],
        [64,7,256,5],
        [128,7,64,5],
        [256,7,64,5],
        [128,7,128,5],
        [128,7,256,5],
        [256,7,128,5],
        [256,7,256,5],
        [256,7,400,5],
        [400,7,256,5],
        [400,7,400,5],
        [400,7,800,5],
        [800,7,400,5],
        [800,7,800,5],
        [64,7,64,7],
        [64,7,128,7],
        [64,7,256,7],
        [128,7,64,7],
        [256,7,64,7],
        [128,7,128,7],
        [128,7,256,7],
        [256,7,128,7],
        [256,7,256,7],
        [256,7,400,7],
        [400,7,256,7],
        [400,7,400,7],
        [400,7,800,7],
        [800,7,400,7],
        [800,7,800,7],
        [64,7,64,9],
        [64,7,128,9],
        [64,7,256,9],
        [128,7,64,9],
        [256,7,64,9],
        [128,7,128,9],
        [128,7,256,9],
        [256,7,128,9],
        [256,7,256,9],
        [256,7,400,9],
        [400,7,256,9],
        [400,7,400,9],
        [400,7,800,9],
        [800,7,400,9],
        [800,7,800,9]]
elif model == 'C2Z':
    model_name = "ConvNet 2layer Z"
    p = [
        [64,9,64,7],
        [64,9,128,7],
        [64,9,256,7],
        [128,9,64,7],
        [256,9,64,7],
        [128,9,128,7],
        [128,9,256,7],
        [256,9,128,7],
        [256,9,256,7],
        [256,9,400,7],
        [400,9,256,7],
        [400,9,400,7],
        [400,9,800,7],
        [800,9,400,7],
        [800,9,800,7],
        [64,9,64,9],
        [64,9,128,9],
        [64,9,256,9],
        [128,9,64,9],
        [256,9,64,9],
        [128,9,128,9],
        [128,9,256,9],
        [256,9,128,9],
        [256,9,256,9],
        [256,9,400,9],
        [400,9,256,9],
        [400,9,400,9],
        [400,9,800,9],
        [800,9,400,9],
        [800,9,800,9],
        [64,11,64,11],
        [64,11,128,11],
        [64,11,256,11],
        [128,11,64,11],
        [256,11,64,11],
        [128,11,128,11],
        [128,11,256,11],
        [256,11,128,11],
        [256,11,256,11],
        [256,11,400,11],
        [400,11,256,11],
        [400,11,400,11],
        [400,11,800,11],
        [800,11,400,11],
        [800,11,800,11],
        [64,13,64,13],
        [64,13,128,13],
        [64,13,256,13],
        [128,13,64,13],
        [256,13,64,13],
        [128,13,128,13],
        [128,13,256,13],
        [256,13,128,13],
        [256,13,256,13],
        [256,13,400,13],
        [400,13,256,13],
        [400,13,400,13],
        [400,13,800,13],
        [800,13,400,13],
        [800,13,800,13]]
elif model == 'C':
    model_name = "ConvNet 3layer "
    p = [
        [200,11,400,11,200,3],
        [200,11,400,11,200,5],
        [200,11,400,11,200,7],
        [200,11,400,11,200,9],
        [200,11,400,11,200,13],
        [200,11,400,11,200,15],
        [200,11,400,3,200,11],
        [200,11,400,5,200,11],
        [200,11,400,7,200,11],
        [200,11,400,9,200,11],
        [200,11,400,13,200,11],
        [200,11,400,15,200,11],
        [200,3,400,11,200,11],
        [200,5,400,11,200,11],
        [200,7,400,11,200,11],
        [200,9,400,11,200,11],
        [200,13,400,11,200,11],
        [200,15,400,11,200,11],
        [200,11,400,3,200,3],
        [200,11,400,5,200,5],
        [200,11,400,7,200,7],
        [200,11,400,9,200,9],
        [200,11,400,13,200,13],
        [200,11,400,15,200,15],
        [200,3,400,3,200,11],
        [200,5,400,5,200,11],
        [200,7,400,7,200,11],
        [200,9,400,9,200,11],
        [200,13,400,13,200,11],
        [200,15,400,15,200,11],
        [200,3,400,11,200,3],
        [200,5,400,11,200,5],
        [200,7,400,11,200,7],
        [200,9,400,11,200,9],
        [200,13,400,11,200,13],
        [200,15,400,11,200,15],
        [200,3,400,3,200,3],
        [200,5,400,5,200,5],
        [200,7,400,7,200,7],
        [200,9,400,9,200,9],
        [200,11,400,11,200,11],
        [200,13,400,13,200,13],
        [200,15,400,15,200,15],
        [200,13,400,13,200,3],
        [200,13,400,13,200,5],
        [200,13,400,13,200,7],
        [200,13,400,13,200,9],
        [200,13,400,13,200,11],
        [200,13,400,13,200,15],
        [200,13,400,3,200,13],
        [200,13,400,5,200,13],
        [200,13,400,7,200,13],
        [200,13,400,9,200,13],
        [200,13,400,11,200,13],
        [200,13,400,15,200,13],
        [200,3,400,13,200,13],
        [200,5,400,13,200,13],
        [200,7,400,13,200,13],
        [200,9,400,13,200,13],
        [200,11,400,13,200,13],
        [200,15,400,13,200,13],
        [200,13,400,3,200,3],
        [200,13,400,5,200,5],
        [200,13,400,7,200,7],
        [200,13,400,9,200,9],
        [200,13,400,11,200,11],
        [200,13,400,15,200,15],
        [200,3,400,3,200,13],
        [200,5,400,5,200,13],
        [200,7,400,7,200,13],
        [200,9,400,9,200,13],
        [200,11,400,11,200,13],
        [200,15,400,15,200,13],
        [200,3,400,13,200,3],
        [200,5,400,13,200,5],
        [200,7,400,13,200,7],
        [200,9,400,13,200,9],
        [200,11,400,13,200,11],
        [200,15,400,13,200,15],
        [200,9,400,9,200,3],
        [200,9,400,9,200,5],
        [200,9,400,9,200,7],
        [200,9,400,9,200,11],
        [200,9,400,9,200,13],
        [200,9,400,9,200,15],
        [200,9,400,3,200,9],
        [200,9,400,5,200,9],
        [200,9,400,7,200,9],
        [200,9,400,11,200,9],
        [200,9,400,13,200,9],
        [200,9,400,15,200,9],
        [200,3,400,9,200,9],
        [200,5,400,9,200,9],
        [200,7,400,9,200,9],
        [200,11,400,9,200,9],
        [200,13,400,9,200,9],
        [200,15,400,9,200,9],
        [200,9,400,3,200,3],
        [200,9,400,5,200,5],
        [200,9,400,7,200,7],
        [200,9,400,11,200,11],
        [200,9,400,13,200,13],
        [200,9,400,15,200,15],
        [200,3,400,3,200,9],
        [200,5,400,5,200,9],
        [200,7,400,7,200,9],
        [200,11,400,11,200,9],
        [200,13,400,13,200,9],
        [200,15,400,15,200,9],
        [200,3,400,9,200,3],
        [200,5,400,9,200,5],
        [200,7,400,9,200,7],
        [200,11,400,9,200,11],
        [200,13,400,9,200,13],
        [200,15,400,9,200,15]]
elif model == 'C3-D1':
    model_name = "3Conv-1Dense layer "
    p = [
        [200,3,400,9,200,9,16],
        [200,3,400,9,200,9,64],
        [200,3,400,9,200,9,128],
        [200,3,400,9,200,9,256],
        [200,3,400,9,200,9,512],
        [200,3,400,9,200,9,1024],
        [200,3,400,9,200,9,2048],
        [200,3,400,9,200,9,4096]]
elif model == 'C3-D2':
    model_name = "3Conv-2Dense layer"
    p = [
        [200,3,400,13,200,13,16,16],
        [200,3,400,13,200,13,64,64],
        [200,3,400,13,200,13,128,128],
        [200,3,400,13,200,13,256,256],
        [200,3,400,13,200,13,512,512],
        [200,3,400,13,200,13,1024,1024],
        [200,3,400,13,200,13,2048,2048],
        [200,3,400,13,200,13,4096,4096],
        [200,3,400,13,200,13,16,64],
        [200,3,400,13,200,13,64,128],
        [200,3,400,13,200,13,128,256],
        [200,3,400,13,200,13,256,512],
        [200,3,400,13,200,13,512,1024],
        [200,3,400,13,200,13,2048,4096],
        [200,3,400,13,200,13,4096,8192],
        [200,3,400,13,200,13,64,16],
        [200,3,400,13,200,13,128,64],
        [200,3,400,13,200,13,256,128],
        [200,3,400,13,200,13,512,256],
        [200,3,400,13,200,13,1024,512],
        [200,3,400,13,200,13,2048,1024],
        [200,3,400,13,200,13,4096,2048],
        [200,3,400,13,200,13,8192,4096],
        [200,3,400,13,200,13,16,16]]
elif model == 'A':
    model_name = "AlexNet "
    p = [
    [96,3,256,5,384,3,384,3,256,3,4096,4096],
    [96,5,256,5,384,3,384,3,256,3,4096,4096],
    [96,7,256,5,384,3,384,3,256,3,4096,4096],
    [96,9,256,5,384,3,384,3,256,3,4096,4096],
    [96,11,256,5,384,3,384,3,256,3,4096,4096],
    [96,13,256,5,384,3,384,3,256,3,4096,4096],
    [96,15,256,5,384,3,384,3,256,3,4096,4096],
    [96,11,256,3,384,3,384,3,256,3,4096,4096],
    [96,11,256,5,384,3,384,3,256,3,4096,4096],
    [96,11,256,7,384,3,384,3,256,3,4096,4096],
    [96,11,256,9,384,3,384,3,256,3,4096,4096],
    [96,11,256,11,384,3,384,3,256,3,4096,4096],
    [96,11,256,5,384,3,384,3,256,3,4096,4096],
    [96,11,256,5,384,5,384,3,256,3,4096,4096],
    [96,11,256,5,384,7,384,3,256,3,4096,4096],
    [96,11,256,5,384,9,384,3,256,3,4096,4096],
    [96,11,256,5,384,11,384,3,256,3,4096,4096],
    [96,11,256,5,384,3,384,3,256,3,4096,4096],
    [96,11,256,5,384,3,384,5,256,3,4096,4096],
    [96,11,256,5,384,3,384,7,256,3,4096,4096],
    [96,11,256,5,384,3,384,9,256,3,4096,4096],
    [96,11,256,5,384,3,384,11,256,3,4096,4096],
    [96,11,256,5,384,3,384,3,256,3,4096,4096],
    [96,11,256,5,384,3,384,3,256,5,4096,4096],
    [96,11,256,5,384,3,384,3,256,7,4096,4096],
    [96,11,256,5,384,3,384,3,256,9,4096,4096],
    [96,11,256,5,384,3,384,3,256,11,4096,4096],
    [96,11,256,5,384,3,384,3,256,3,4096,4096],
    [96,11,256,5,384,3,384,3,256,3,2048,4096],
    [96,11,256,5,384,3,384,3,256,3,1024,4096],
    [96,11,256,5,384,3,384,3,256,3,512,4096],
    [96,11,256,5,384,3,384,3,256,3,256,4096],
    [96,11,256,5,384,3,384,3,256,3,128,4096],
    [96,11,256,5,384,3,384,3,256,3,64,4096],
    [96,11,256,5,384,3,384,3,256,3,4096,2048],
    [96,11,256,5,384,3,384,3,256,3,4096,1024],
    [96,11,256,5,384,3,384,3,256,3,4096,512],
    [96,11,256,5,384,3,384,3,256,3,4096,256],
    [96,11,256,5,384,3,384,3,256,3,4096,128],
    [96,11,256,5,384,3,384,3,256,3,4096,64],
    [96,11,256,5,384,3,384,3,256,3,2048,2048],
    [96,11,256,5,384,3,384,3,256,3,2048,1024],
    [96,11,256,5,384,3,384,3,256,3,1024,2048],
    [96,11,256,5,384,3,384,3,256,3,1024,1024],
    [96,11,256,5,384,3,384,3,256,3,1024,512],
    [96,11,256,5,384,3,384,3,256,3,512,1024],
    [96,11,256,5,384,3,384,3,256,3,512,512],
    [96,11,256,5,384,3,384,3,256,3,512,256],
    [96,11,256,5,384,3,384,3,256,3,256,512],
    [96,11,256,5,384,3,384,3,256,3,256,256],
    [96,11,256,5,384,3,384,3,256,3,256,128],
    [96,11,256,5,384,3,384,3,256,3,128,256],
    [96,11,256,5,384,3,384,3,256,3,128,128],
    [96,11,256,5,384,3,384,3,256,3,128,64],
    [96,11,256,5,384,3,384,3,256,3,64,128],
    [96,11,256,5,384,3,384,3,256,3,64,64],
    [96,11,256,5,384,3,384,3,256,3,64,32],
    [96,11,256,5,384,3,384,3,256,3,32,64],
    [96,11,256,5,384,3,384,3,256,3,32,32]]
elif model == 'L':
    model_name = "LeNet "
    p = [
        [10,3]]
elif model == 'V':
    model_name = "VGGNet "
    p = [
    [64,3,128,3,256,3,512,3,512,3,4096,4096],
    [64,5,128,3,256,3,512,3,512,3,4096,4096],
    [64,7,128,3,256,3,512,3,512,3,4096,4096],
    [64,9,128,3,256,3,512,3,512,3,4096,4096],
    [64,11,128,3,256,3,512,3,512,3,4096,4096],
    [64,13,128,3,256,3,512,3,512,3,4096,4096],
    [64,15,128,3,256,3,512,3,512,3,4096,4096],
    [64,3,128,3,256,3,512,3,512,3,4096,4096],
    [64,3,128,5,256,3,512,3,512,3,4096,4096],
    [64,3,128,7,256,3,512,3,512,3,4096,4096],
    [64,3,128,9,256,3,512,3,512,3,4096,4096],
    [64,3,128,11,256,3,512,3,512,3,4096,4096],
    [64,3,128,13,256,3,512,3,512,3,4096,4096],
    [64,3,128,15,256,3,512,3,512,3,4096,4096],
    [64,3,128,3,256,3,512,3,512,3,4096,4096],
    [64,3,128,3,256,5,512,3,512,3,4096,4096],
    [64,3,128,3,256,7,512,3,512,3,4096,4096],
    [64,3,128,3,256,9,512,3,512,3,4096,4096],
    [64,3,128,3,256,11,512,3,512,3,4096,4096],
    [64,3,128,3,256,13,512,3,512,3,4096,4096],
    [64,3,128,3,256,15,512,3,512,3,4096,4096],
    [64,3,128,3,256,3,512,3,512,3,4096,4096],
    [64,3,128,3,256,3,512,5,512,3,4096,4096],
    [64,3,128,3,256,3,512,7,512,3,4096,4096],
    [64,3,128,3,256,3,512,9,512,3,4096,4096],
    [64,3,128,3,256,3,512,11,512,3,4096,4096],
    [64,3,128,3,256,3,512,13,512,3,4096,4096],
    [64,3,128,3,256,3,512,15,512,3,4096,4096],
    [64,3,128,3,256,3,512,3,512,3,4096,4096],
    [64,3,128,3,256,3,512,3,512,5,4096,4096],
    [64,3,128,3,256,3,512,3,512,7,4096,4096],
    [64,3,128,3,256,3,512,3,512,9,4096,4096],
    [64,3,128,3,256,3,512,3,512,11,4096,4096],
    [64,3,128,3,256,3,512,3,512,13,4096,4096],
    [64,3,128,3,256,3,512,3,512,15,4096,4096],
    [64,3,128,3,256,3,512,3,512,3,4096,4096],
    [64,3,128,3,256,3,512,3,512,3,4096,2048],
    [64,3,128,3,256,3,512,3,512,3,4096,1024],
    [64,3,128,3,256,3,512,3,512,3,4096,512],
    [64,3,128,3,256,3,512,3,512,3,4096,256],
    [64,3,128,3,256,3,512,3,512,3,4096,128],
    [64,3,128,3,256,3,512,3,512,3,4096,64],
    [64,3,128,3,256,3,512,3,512,3,4096,32],
    [64,3,128,3,256,3,512,3,512,3,2048,4096],
    [64,3,128,3,256,3,512,3,512,3,1024,4096],
    [64,3,128,3,256,3,512,3,512,3,512,4096],
    [64,3,128,3,256,3,512,3,512,3,256,4096],
    [64,3,128,3,256,3,512,3,512,3,128,4096],
    [64,3,128,3,256,3,512,3,512,3,64,4096],
    [64,3,128,3,256,3,512,3,512,3,32,4096],
    [64,3,128,3,256,3,512,3,512,3,2048,2048],
    [64,3,128,3,256,3,512,3,512,3,1024,1024],
    [64,3,128,3,256,3,512,3,512,3,512,512],
    [64,3,128,3,256,3,512,3,512,3,256,256],
    [64,3,128,3,256,3,512,3,512,3,128,128],
    [64,3,128,3,256,3,512,3,512,3,64,64],
    [64,3,128,3,256,3,512,3,512,3,32,32],
    [64,3,128,3,256,3,512,3,512,3,16,16]]
elif model == 'R':
    model_name = "ResNet "
    p = [
        # [64,3,64,3,128,3,256,3,512,3,1000],
        # [64,5,64,3,128,3,256,3,512,3,1000],
        # [64,7,64,3,128,3,256,3,512,3,1000],
        # [64,9,64,3,128,3,256,3,512,3,1000],
        # [64,11,64,3,128,3,256,3,512,3,1000],
        # [64,13,64,3,128,3,256,3,512,3,1000],
        # [64,15,64,3,128,3,256,3,512,3,1000],

        [16,9,64,3,128,3,256,3,512,3,1000],
        [64,9,64,3,128,3,256,3,512,3,1000],
        [128,9,64,3,128,3,256,3,512,3,1000],
        [256,9,64,3,128,3,256,3,512,3,1000],
        [512,9,64,3,128,3,256,3,512,3,1000],
        [1024,9,64,3,128,3,256,3,512,3,1000],
        [2048,9,64,3,128,3,256,3,512,3,1000],
        [4096,9,64,3,128,3,256,3,512,3,1000],
        [64,9,64,3,128,3,256,3,512,3,16],
        [64,9,64,3,128,3,256,3,512,3,64],
        [64,9,64,3,128,3,256,3,512,3,128],
        [64,9,64,3,128,3,256,3,512,3,256],
        [64,9,64,3,128,3,256,3,512,3,512],
        [64,9,64,3,128,3,256,3,512,3,1024],
        [64,9,64,3,128,3,256,3,512,3,2048],
        [64,9,64,3,128,3,256,3,512,3,4096],

        [64,9,64,5,128,3,256,3,512,3,1000],
        [64,9,64,7,128,3,256,3,512,3,1000],
        [64,9,64,9,128,3,256,3,512,3,1000],
        [64,9,64,11,128,3,256,3,512,3,1000],
        [64,9,64,13,128,3,256,3,512,3,1000],
        [64,9,64,15,128,3,256,3,512,3,1000],
        [64,9,64,3,128,3,256,3,512,3,1000],
        [64,9,64,3,128,5,256,3,512,3,1000],
        [64,9,64,3,128,7,256,3,512,3,1000],
        [64,9,64,3,128,9,256,3,512,3,1000],
        [64,9,64,3,128,11,256,3,512,3,1000],
        [64,9,64,3,128,13,256,3,512,3,1000],
        [64,9,64,3,128,15,256,3,512,3,1000],
        [64,9,64,3,128,3,256,3,512,3,1000],
        [64,9,64,3,128,3,256,5,512,3,1000],
        [64,9,64,3,128,3,256,7,512,3,1000],
        [64,9,64,3,128,3,256,9,512,3,1000],
        [64,9,64,3,128,3,256,11,512,3,1000],
        [64,9,64,3,128,3,256,13,512,3,1000],
        [64,9,64,3,128,3,256,15,512,3,1000],
        [64,9,64,3,128,3,256,3,512,3,1000],
        [64,9,64,3,128,3,256,3,512,5,1000],
        [64,9,64,3,128,3,256,3,512,7,1000],
        [64,9,64,3,128,3,256,3,512,9,1000],
        [64,9,64,3,128,3,256,3,512,11,1000],
        [64,9,64,3,128,3,256,3,512,13,1000],
        [64,9,64,3,128,3,256,3,512,15,1000],

        [64,3,64,3,128,3,256,3,512,3,1000],
        [64,5,64,5,128,5,256,5,512,5,1000],
        [64,7,64,7,128,7,256,7,512,7,1000],
        [64,9,64,9,128,9,256,9,512,9,1000],
        [64,11,64,11,128,11,256,11,512,11,1000],
        [64,13,64,13,128,13,256,13,512,13,1000],
        [64,15,64,15,128,15,256,15,512,15,1000],

        [64,9,64,3,128,3,256,3,512,3,1000],
        [64,9,64,5,128,5,256,5,512,5,1000],
        [64,9,64,7,128,7,256,7,512,7,1000],
        [64,9,64,9,128,9,256,9,512,9,1000],
        [64,9,64,11,128,11,256,11,512,11,1000],
        [64,9,64,13,128,13,256,13,512,13,1000],
        [64,9,64,15,128,15,256,15,512,15,1000]]
    
for i in range(1):
    project_name = "people_counting==" + model_name
    wandb.init(project=project_name,reinit=True)
    num = i
    config = wandb.config
    config.learning_rate = learning_rate
    config.epochs = epoch

    # wandb.init(project="people_counting==AlexNet",reinit=True)
    # model selection
    if model == 'F':
        print("model name : FCNet ")
        i = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(64,2)),
            tf.keras.layers.Dense(200,activation=tf.nn.relu),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(400,activation=tf.nn.relu),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(200,activation=tf.nn.relu),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(6,activation='softmax')
        ])
    elif model == 'C':
        print("model name : ConvNet")
        i = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(200,3, activation=tf.nn.relu,input_shape=(64,2),padding='same'),
            # tf.keras.layers.LayerNormalization(axis=-1),
            # tf.keras.layers.BatchNormalization(axis=-1),
            # tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv1D(400,13, activation=tf.nn.relu,padding='same'),
            # tf.keras.layers.LayerNormalization(axis=-1),
            # tf.keras.layers.BatchNormalization(axis=-1),
            # tf.keras.layers.Dropout(0.3),
            # tf.keras.layers.MaxPooling1D(2,padding='same'),
            tf.keras.layers.Conv1D(200,13, activation=tf.nn.relu,padding='same'),
            # tf.keras.layers.LayerNormalization(axis=-1),
            # tf.keras.layers.BatchNormalization(axis=-1),
            # tf.keras.layers.Dropout(0.3),
            # tf.keras.layers.MaxPooling1D(2,padding='same'),
            # tf.keras.layers.Conv1D(512,3, activation=tf.nn.relu,padding='same'),
            # tf.keras.layers.LayerNormalization(axis=-1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(6, activation='softmax')
        ])
    elif model == 'A':
        print("model name : AlexNet")
        i = tf.keras.Sequential([
            # layer1
            tf.keras.layers.Conv1D(96,11, activation=tf.nn.relu,input_shape=(64,2),padding='same'),
            # tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.AveragePooling1D(2,padding='same'),
            # layer2
            tf.keras.layers.Conv1D(256,5, activation=tf.nn.relu,padding='same'),
            # tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.AveragePooling1D(2,padding='same'),
            # layer3
            tf.keras.layers.ZeroPadding1D(1),
            tf.keras.layers.Conv1D(384,3, activation=tf.nn.relu,padding='same'),
            # tf.keras.layers.BatchNormalization(axis=-1),
            # layer4
            tf.keras.layers.ZeroPadding1D(1),
            tf.keras.layers.Conv1D(384,7, activation=tf.nn.relu,padding='same'),
            # tf.keras.layers.BatchNormalization(axis=-1),
            # layer5
            tf.keras.layers.ZeroPadding1D(1),
            tf.keras.layers.Conv1D(256,3, activation=tf.nn.relu,padding='same'),
            # tf.keras.layers.BatchNormalization(axis=-1),
            # layer6
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation='relu'),
            # tf.keras.layers.Dropout(0.1),
            # layer7
            tf.keras.layers.Dense(4096, activation='relu'),
            # tf.keras.layers.Dropout(0.1),
            # layer8
            tf.keras.layers.Dense(6, activation='softmax')
        ])
    elif model == 'L':
        print("model name : LeNet")
        i = tf.keras.Sequential([
            # layer1
            tf.keras.layers.Conv1D(6,5, activation=tf.nn.relu,input_shape=(64,2),padding='same'),
            # layer2
            tf.keras.layers.AveragePooling1D(2,padding='same'),
            # layer3
            tf.keras.layers.Conv1D(16,5, activation=tf.nn.relu,padding='same'),
            # layer4
            tf.keras.layers.AveragePooling1D(2,padding='same'),
            # layer5
            tf.keras.layers.Conv1D(120,5, activation=tf.nn.relu,input_shape=(64,2),padding='same'),
            # layer6
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(840, activation='relu'),
            # layer7
            tf.keras.layers.Dense(6, activation='softmax'),
        ])
    elif model == 'V':
        print("model name : VGGNet")
        i = tf.keras.Sequential([
            # layer1~3
            tf.keras.layers.Conv1D(p[i][0],p[i][1], activation=tf.nn.relu,input_shape=(64,2),padding='same'),
            tf.keras.layers.Conv1D(p[i][0],p[i][1], activation=tf.nn.relu,padding='same'),
            tf.keras.layers.AveragePooling1D(2,padding='same'),
            # layer4~6
            tf.keras.layers.Conv1D(p[i][2],p[i][3], activation=tf.nn.relu,padding='same'),
            tf.keras.layers.Conv1D(p[i][2],p[i][3], activation=tf.nn.relu,padding='same'),
            tf.keras.layers.AveragePooling1D(2,padding='same'),
            # layer7~11
            tf.keras.layers.Conv1D(p[i][4],p[i][5], activation=tf.nn.relu,padding='same'),
            tf.keras.layers.Conv1D(p[i][4],p[i][5], activation=tf.nn.relu,padding='same'),
            tf.keras.layers.Conv1D(p[i][4],p[i][5], activation=tf.nn.relu,padding='same'),
            tf.keras.layers.Conv1D(p[i][4],p[i][5], activation=tf.nn.relu,padding='same'),
            tf.keras.layers.AveragePooling1D(2,padding='same'),
            # layer12~16
            tf.keras.layers.Conv1D(p[i][6],p[i][7], activation=tf.nn.relu,padding='same'),
            tf.keras.layers.Conv1D(p[i][6],p[i][7], activation=tf.nn.relu,padding='same'),
            tf.keras.layers.Conv1D(p[i][6],p[i][7], activation=tf.nn.relu,padding='same'),
            tf.keras.layers.Conv1D(p[i][6],p[i][7], activation=tf.nn.relu,padding='same'),
            tf.keras.layers.AveragePooling1D(2,padding='same'),
            # layer17~21
            tf.keras.layers.Conv1D(p[i][8],p[i][9], activation=tf.nn.relu,padding='same'),
            tf.keras.layers.Conv1D(p[i][8],p[i][9], activation=tf.nn.relu,padding='same'),
            tf.keras.layers.Conv1D(p[i][8],p[i][9], activation=tf.nn.relu,padding='same'),
            tf.keras.layers.Conv1D(p[i][8],p[i][9], activation=tf.nn.relu,padding='same'),
            tf.keras.layers.AveragePooling1D(2,padding='same'),
            # layer22~24
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(p[i][10], activation=tf.nn.relu),
            tf.keras.layers.Dense(p[i][11], activation=tf.nn.relu),
            tf.keras.layers.Dense(6, activation='softmax'),
        ])
    elif model == 'R':
        print("model name : ResNet")
        def identity_block(x, filters, kernel_size):
            shortcut = x
            
            x = tf.keras.layers.Conv1D(filters,kernel_size,padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            
            x = tf.keras.layers.Conv1D(filters,kernel_size,padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            
            x = tf.keras.layers.Add()([x, shortcut])
            x = tf.keras.layers.Activation('relu')(x)
            
            return x
        
        def conv_block(x, filters, kernel_size):
            shortcut = x
            
            x = tf.keras.layers.Conv1D(filters,kernel_size,padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            
            x = tf.keras.layers.Conv1D(filters,kernel_size,padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            
            shortcut = tf.keras.layers.Conv1D(filters,kernel_size,padding='same')(shortcut)
            shortcut = tf.keras.layers.BatchNormalization()(shortcut)

            x = tf.keras.layers.Add()([x, shortcut])
            x = tf.keras.layers.Activation('relu')(x)
            
            return x

        def ResNet18(input_shape = (64,2), classes = 6):
            x_input = tf.keras.layers.Input(input_shape)
            x = x_input
            # blockA
            x = tf.keras.layers.Conv1D(4096,9,2,activation=tf.nn.relu,padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPooling1D(2,padding='same')(x)
            # blockB
            x = conv_block(x,64,3)
            x = identity_block(x,64,3)
            x = tf.keras.layers.MaxPooling1D(2,padding='same')(x)
            # blockC
            x = conv_block(x,128,3)
            x = identity_block(x,128,3)
            x = tf.keras.layers.MaxPooling1D(2,padding='same')(x)
            # blockD
            x = conv_block(x,256,3)
            x = identity_block(x,256,3)
            x = tf.keras.layers.MaxPooling1D(2,padding='same')(x)
            # blockE
            x = conv_block(x,512,3)
            x = identity_block(x,512,3)
            x = tf.keras.layers.MaxPooling1D(2,padding='same')(x)
            # blockF
            x = tf.keras.layers.AveragePooling1D(2,padding='same')(x)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(1000,activation=tf.nn.relu)(x)
            x = tf.keras.layers.Dense(classes,activation=tf.nn.softmax)(x)
            
            model = tf.keras.models.Model(inputs = x_input, outputs = x, name = "ResNet18")
            return model
        i = ResNet18()
        # i.build(input_shape = (None,64,2))
    name = model_name + f'{num}' 
    # + ", start_lr = " + f'{strat_lr}' + ", epoch_drop = " + f'{epoch_drop}' ", drop_ratio = " + f'{learning_rate}'
    
    wandb.run.name = name
    
    i.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    print(name)
    print(p[num])
    # print(i.summary())
    # i.save('./FCNet')
    history = i.fit(X_train, y_train, epochs=epoch,batch_size=32,
                validation_data=(X_val, y_val),
                callbacks=[lr_scheduler, earlystopping, WandbCallback()])
    
    i.save('./ResNet_best4.h5')
    # print(i.get_weights())
    acc = i.evaluate(X_test, y_test)
    result.append((p[num],acc[1]))
    # wandb.run.save()


# save_name = model_name + "result.npy"
# np.save(save_name,result)