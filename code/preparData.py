import librosa
import scipy.io
import numpy as np
import torch
import librosa.display
import matplotlib.pyplot as plt
import os
import h5py
import random

'''
Written by KKl in 2020-12-1

This file is to extract feature and save it as HDF5 file
'''


def save_h5(h5file, data_type, data, name, dtype=np.float32):
    '''
    save and add data to hdf5 file
    :param h5file: an opened h5 file
    :param data:
    :param name: dataset name
    :param dtype: np type, such as np.float32
    '''
    shape_list=list(data.shape)
    if not h5file[data_type].__contains__(name):
        shape_list[0]=None #设置数组的第一个维度是0
        dataset = h5file[data_type].create_dataset(name, data=data, maxshape=tuple(shape_list), chunks=True, dtype=dtype)
    else:
        dataset = h5file[data_type][name]
    len_old=dataset.shape[0]
    len_new=len_old+data.shape[0]
    shape_list[0]=len_new
    dataset.resize(tuple(shape_list)) #修改数组的第一个维度
    dataset[len_old:len_new] = data  #存入新的文件

def file_name(path, postfix='.wav'):
    F = []
    for root, dirs, files in os.walk(path):
        for file in files:
            #print file.decode('gbk')    #文件名中有中文字符时转码
            if os.path.splitext(file)[1] == postfix:
                F.append(file) #将所有的文件名添加到L列表中
    return F   # 返回L列表

path = r"C:\Users\...\语音信号处理\Recorded_data"
hdf5_path = r"C:\Users\...\语音信号处理\data.hdf5"
frame_point = 160000 # 2s
# samplerate = 16000
n_mfcc = 40

wav_file_list = file_name(path, postfix='.wav')
data = []
for f in wav_file_list:
    file_path = os.path.join(path,f)
    y,sr= librosa.load(file_path, sr=None)
    data.append(y)
data = np.concatenate(data)
print('data shape:',data.shape)


mat_file_list = file_name(path, postfix='.mat')
label = []
for f in mat_file_list:
    file_path = os.path.join(path,f)
    y = scipy.io.loadmat(file_path)
    y = y['y_label'].squeeze()
    label.append(y)
label = np.concatenate(label)
print('label shape:',label.shape)

FEATURE = []
LABEL = []
length = data.shape[0]//frame_point
print('length', length)

for i in range(length-1):
    y = data[i*frame_point:i*frame_point+frame_point]
    l = label[i*frame_point:i*frame_point+frame_point]
    mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=n_mfcc).transpose()
    # mfcc = librosa.power_to_db(mfcc)
    # mfcc = (mfcc - np.expand_dims(np.mean(mfcc, axis=1), axis=1))
    # mfcc = mfcc/np.expand_dims(np.std(mfcc, axis=1), axis=1)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length = 1024, hop_length = 512, center = True).transpose()


    label_stft = librosa.stft(l.astype(np.float), n_fft=1024,hop_length = 512, center = True)
    label_stft = abs(label_stft)//400
    label_stft = label_stft.transpose()
    label_stft = np.expand_dims(label_stft[:, 0], axis=1)
    feature = np.concatenate([mfcc, zcr], axis=1)
    FEATURE.append(feature)
    LABEL.append(label_stft)

    if (i+1)%100 == 0 or i==length-2:
        num = i%100+1
        print(i, num,'writing HDF5')
        idx = list(range(num))
        random.shuffle(idx)
        f = [FEATURE[i] for i in idx]
        l = [LABEL[i] for i in idx]

        f = np.stack(f, axis=0)
        l = np.stack(l, axis=0)
        with h5py.File(hdf5_path, 'a') as hf:
            if 'train' not in hf.keys():
                hf.create_group('train')
            save_h5(hf, 'train', f[0:int(num*0.7)], 'feature', dtype=np.float64)
            save_h5(hf, 'train', l[0:int(num*0.7)], 'label', dtype=np.float64)

            if 'test' not in hf.keys():
                hf.create_group('test')
            save_h5(hf, 'test', f[int(num*0.7):int(num*0.85)], 'feature', dtype=np.float64)
            save_h5(hf, 'test', l[int(num*0.7):int(num*0.85)], 'label', dtype=np.float64)

            if 'valid' not in hf.keys():
                    hf.create_group('valid')
            save_h5(hf, 'valid', f[int(num*0.85):], 'feature', dtype=np.float64)
            save_h5(hf, 'valid', l[int(num*0.85):], 'label', dtype=np.float64)

        FEATURE = []
        LABEL = []
print(i)
