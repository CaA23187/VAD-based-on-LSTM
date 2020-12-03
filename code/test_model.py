import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from GELu import GELu
from My_Dataset import MyDataset
from pytorchtools import EarlyStopping
from LSTM import LSTM


if __name__ == '__main__':
    BATCH_SIZE = 10000

    hdf5_dir = hdf5_dir = r'C:\Users\...\语音信号处理\data.hdf5'
    test_data = MyDataset(hdf5_dir, 'test')
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

    torch.set_default_dtype(torch.float64)
    net = LSTM()
    loss_func = nn.MSELoss()
    net.load_state_dict(torch.load('./VAD.pkl'))

    net.eval() # prep model for evaluation
    with torch.no_grad():
        for feature, label in test_loader:
            feature = feature
            label = label

            output = net(feature)
            loss = loss_func(output.squeeze(), label.squeeze())
            print('loss',loss.item())
            output = output.numpy()
            label = label.numpy()

            result = (np.sign(output-0.5)+2)//2
            acc = 1-np.sum((result-label)**2)/(label.shape[0]*label.shape[1]*label.shape[2])
            print('acc = ',acc)
            plt.figure()
            plt.plot(result[15,:,0])
            plt.plot(label[15,:,0])
            plt.plot(output[15,:,0])
            plt.legend(['result','label','output'])

            plt.figure()
            plt.plot(result[16,:,0])
            plt.plot(label[16,:,0])
            plt.plot(output[16,:,0])
            plt.legend(['result','label','output'])

            plt.figure()
            plt.plot(result[59,:,0])
            plt.plot(label[59,:,0])
            plt.plot(output[59,:,0])
            plt.legend(['result','label','output'])

            plt.show()
            # record validation loss



