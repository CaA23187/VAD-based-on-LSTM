import h5py
import torch
from torch.utils.data import Dataset
import numpy as np



class MyDataset(Dataset):
    def __init__(self, hdf5path, type):
        assert type in ['train', 'valid','test']
        with h5py.File(hdf5path, 'r') as hf:
            self.feature = hf[type]['feature'][:]
            self.label = hf[type]['label'][:]
        self.len = len(self.feature)

    def __getitem__(self, i):
        index = i % self.len
        feature = self.feature[index]
        label = self.label[index]
        feature, label = self.data_preproccess((feature, label))

        return feature, label

    def __len__(self):
        return self.len

    def data_preproccess(self, data):
        """
        数据预处理
        :param data:
        :return:
        """
        feature, label = data

        feature = torch.from_numpy(feature)
        label = torch.from_numpy(label)


        return feature, label

