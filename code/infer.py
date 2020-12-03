import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import time
import os
import csv
import torch.nn.init as init
import matplotlib.pyplot as plt

from GELu import GELu
from My_Dataset import MyDataset
from pytorchtools import EarlyStopping
from train import LSTM

def real_to_complex(pd_abs_x, gt_x):
    """Recover pred spectrogram's phase from ground truth's phase.

    Args:
      pd_abs_x: 2d array, (n_time, n_freq)
      gt_x: 2d complex array, (n_time, n_freq)

    Returns:
      2d complex array, (n_time, n_freq)
    """
    theta = np.angle(gt_x)
    cmplx = pd_abs_x * np.exp(1j * theta)
    return cmplx

if __name__ == '__main__':
    wav_file = r'C:\Users\....\语音信号处理\Recorded_data\bus_stop.wav'
    wav, sr = librosa.load(wav_file, sr=None)
    mfcc = librosa.feature.mfcc(wav, sr=sr, n_mfcc=26).transpose()
    zcr = librosa.feature.zero_crossing_rate(wav, frame_length = 1024, hop_length = 512, center = True).transpose()

    print(mfcc.shape, zcr.shape)
    feature = np.concatenate([mfcc,zcr],axis=1)
    print(feature.shape)
    feature = torch.from_numpy(np.expand_dims(feature, axis=0))

    torch.set_default_dtype(torch.float64)
    net = LSTM()
    loss_func = nn.MSELoss()
    net.load_state_dict(torch.load('./VAD.pkl'))

    net.eval() # prep model for evaluation
    with torch.no_grad():
        output = net(feature)
        # loss = loss_func(output.squeeze(), label.squeeze())
        # print('loss',loss.item())
        output = output.numpy().squeeze()
        output = (np.sign(output-0.5)+2)//2
        print(output.shape)
        plt.figure()
        plt.plot(output)

        output = output*512
        output = np.expand_dims(output, axis=1)
        output = np.concatenate([output,  np.zeros([output.shape[0], 512])], axis=1).transpose()
        # output = real_to_complex(output, output+0j)
        print(output.shape)
        output = librosa.istft(output.astype(np.complex), hop_length=512)
        plt.figure()
        plt.plot(output)
        plt.show()
        # label = label.numpy()

        # result = (np.sign(output-0.5)+2)//2
        # acc = 1-np.sum((result-label)**2)/(label.shape[0]*label.shape[1]*label.shape[2])
        # print(acc)
        # plt.figure()
        # plt.plot(result[15,:,0])
        # plt.plot(label[15,:,0])
        # plt.plot(output[15,:,0])
        # plt.legend(['result','label','output'])
        #
        # plt.figure()
        # plt.plot(result[59,:,0])
        # plt.plot(label[59,:,0])
        # plt.plot(output[59,:,0])
        # plt.legend(['result','label','output'])
        #
        # plt.show()
