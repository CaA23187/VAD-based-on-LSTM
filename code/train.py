import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import time
import os
import csv
import matplotlib.pyplot as plt

from GELu import GELu
from My_Dataset import MyDataset
from pytorchtools import EarlyStopping
from LSTM import LSTM

'''
Written by KKL on 2020-12-1 

This file is used to train LSTM
'''


def train_model(model, DEVICE, patience, n_epochs, csv_record=False):
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    t1 = time.time()
    for epoch in range(1, n_epochs + 1):
        ###################
        # train the model #
        ###################
        model.train() # prep model for training
        for step, (feature, label) in enumerate(train_loader, 1):
            feature = feature.to(DEVICE)
            label = label.to(DEVICE).squeeze()
            # print(feature.size(), label.size())

            optimizer.zero_grad()
            output = model(feature).squeeze()
            # print(output.size(), label.size())

            loss = loss_func(output, label)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        ######################
        # test the model #
        ######################
        model.eval() # prep model for evaluation
        with torch.no_grad():
            for feature, label in valid_loader:
                feature = feature.to(DEVICE)
                label = label.to(DEVICE)

                output = model(feature)
                loss = loss_func(output.squeeze(), label.squeeze())
                # record validation loss
                valid_losses.append(loss.item())

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(n_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}'+ f'| Using time: {time.time()-t1:.5f}')
        t1 = time.time()

        print(print_msg)
        if csv_record==True:
            with open(train_log_dir, "a", newline="") as train_log:
                writer = csv.writer(train_log)
                writer.writerow([epoch, train_loss])
            with open(valid_log_dir, "a", newline="") as test_log:
                writer = csv.writer(test_log)
                writer.writerow([epoch, valid_loss])


        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))
    return  model, avg_train_losses, avg_valid_losses

if __name__ == '__main__':
    # Hyper Parameters
    EPOCH = 1000
    # BATCH_SIZE = 16
    BATCH_SIZE = 64 # 等下试试16
    LR = 0.001
    patience = 100
    csv_record = True

    # whether use multti GPUs
    MultiGPU = False
    torch.set_default_dtype(torch.float64)
    # torch.backends.cudnn.enabled = False
    print('Epoch = ', EPOCH, '|Batch size = ', BATCH_SIZE, '|Learning rate =', LR)
    if MultiGPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch.cuda.set_device(0)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE=', DEVICE, "| PyTorch", torch.__version__, '| CUDA version ', torch.version.cuda, '| cudnn version', torch.backends.cudnn.version())



    cPath = os.getcwd()  # current path
    hdf5_dir = hdf5_dir = r'C:\Users\...\语音信号处理\data.hdf5'
    train_data = MyDataset(hdf5_dir, 'train')
    valid_data = MyDataset(hdf5_dir, 'valid')
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True,)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, shuffle=False)
    train_log_dir = os.path.join(r'C:\Users\...\语音信号处理\train_log.csv')
    valid_log_dir = os.path.join(r'C:\Users\...\语音信号处理\valid_log.csv')
    print('train data len:',train_data.__len__())

    # log file
    with open(train_log_dir, "w", newline="") as train_log:
        writer = csv.writer(train_log)
        writer.writerow(['epoch', 'loss'])
    with open(valid_log_dir, "w", newline="") as valid_log:
        writer = csv.writer(valid_log)
        writer.writerow(['epoch', 'loss'])

    net = LSTM().to(DEVICE)
    print(net, '\n\n------------------training start-----------------')
    # net.load_state_dict(torch.load('./workspace/'+model_name))
    # optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=0.001)
    loss_func = nn.MSELoss()

    #--------------- training -----------------------
    net, train_loss, valid_loss = train_model(net, DEVICE, patience, EPOCH, csv_record)
    print('---------------result------')
    print('train_loss:',train_loss[-1],'valid_loss:',valid_loss[-1])

    torch.save(net.state_dict(), './VAD.pkl')
    print('save model successfully')
