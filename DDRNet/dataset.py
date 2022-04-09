import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import h5py
from augment import transform

class LandslideDataSet_aug(data.Dataset):
    def __init__(self, data_dir, set='labeled'):
        # self.mean = [-0.4914, -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644, 0.0802, 0.3000, 0.4082, 0.0823, 0.0516,
        #              0.3338, 0.7819]
        self.mean = [0.9257, 0.9227, 0.9541, 0.9596, 1.0228, 1.0426, 1.0358, 1.0468, 1.1699,
        1.1736, 1.0495, 1.0370, 1.2511, 1.6495]
        # self.std = [0.9325, 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354, 0.8491, 0.9061, 1.6072, 0.8848, 0.9232,
        #             0.9018, 1.2913]
        self.std = [0.1410, 0.2207, 0.3184, 0.5724, 0.4601, 0.4465, 0.4651, 0.4948, 0.5133,
        0.6836, 0.5323, 0.6628, 0.6784, 1.0727]
        self.set = set

        if set == 'labeled':
            self.img = os.listdir(data_dir+'\\'+'img')
            self.label = os.listdir(data_dir + '\\' + 'mask')
            self.img = [data_dir+'\\'+'img'+'\\'+i for i in self.img]
            self.label = [data_dir + '\\' + 'mask' + '\\' + i for i in self.label]

        elif set == 'unlabeled':
            self.img = os.listdir(data_dir+'\\'+'img')
            self.img = [data_dir+'\\'+'img'+'\\'+i for i in self.img]
    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        if self.set == 'labeled':
            img_file = self.img[index]
            label_file = self.label[index]
            img_name,label_name = img_file.split('\\')[-1].replace('image','mask'),label_file.split('\\')[-1]
            if img_name != label_name:
                print(img_name,label_name)
                return np.zeros((14,128,128)),np.zeros((128,128))

            with h5py.File(img_file, 'r') as hf:
                image = hf['img'][:]
            with h5py.File(label_file, 'r') as hf:
                label = hf['mask'][:]
            image = np.asarray(image, np.float32)
            label = np.asarray(label, np.float32)
            image, label = transform(image, label)

            image = image.transpose((-1, 0, 1))
            for i in range(len(self.mean)):
                image[i, :, :] -= self.mean[i]
                image[i, :, :] /= self.std[i]

            return image.copy(), label.copy()
        else:
            img_file = self.img[index]
            name = img_file.split('\\')[-1]
            with h5py.File(img_file, 'r') as hf:
                image = hf['img'][:]
            image = np.asarray(image, np.float32)
            image = image.transpose((-1, 0, 1))
            for i in range(len(self.mean)):
                image[i, :, :] -= self.mean[i]
                image[i, :, :] /= self.std[i]
            return image.copy(),name

class LandslideDataSet(data.Dataset):
    def __init__(self, data_dir, set='labeled'):
        # self.mean = [-0.4914, -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644, 0.0802, 0.3000, 0.4082, 0.0823, 0.0516,
        #              0.3338, 0.7819]
        self.mean = [0.9257, 0.9227, 0.9541, 0.9596, 1.0228, 1.0426, 1.0358, 1.0468, 1.1699,
        1.1736, 1.0495, 1.0370, 1.2511, 1.6495]
        # self.std = [0.9325, 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354, 0.8491, 0.9061, 1.6072, 0.8848, 0.9232,
        #             0.9018, 1.2913]
        self.std = [0.1410, 0.2207, 0.3184, 0.5724, 0.4601, 0.4465, 0.4651, 0.4948, 0.5133,
        0.6836, 0.5323, 0.6628, 0.6784, 1.0727]
        self.set = set

        if set == 'labeled':
            self.img = os.listdir(data_dir+'\\'+'img')
            self.label = os.listdir(data_dir + '\\' + 'mask')
            self.img = [data_dir+'\\'+'img'+'\\'+i for i in self.img]
            self.label = [data_dir + '\\' + 'mask' + '\\' + i for i in self.label]

        elif set == 'unlabeled':
            self.img = os.listdir(data_dir+'\\'+'img')
            self.img = [data_dir+'\\'+'img'+'\\'+i for i in self.img]
    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        if self.set == 'labeled':
            img_file = self.img[index]
            label_file = self.label[index]
            img_name,label_name = img_file.split('\\')[-1].replace('image','mask'),label_file.split('\\')[-1]
            if img_name != label_name:
                print(img_name,label_name)
                return np.zeros((14,128,128)),np.zeros((128,128))

            with h5py.File(img_file, 'r') as hf:
                image = hf['img'][:]
            with h5py.File(label_file, 'r') as hf:
                label = hf['mask'][:]
            image = np.asarray(image, np.float32)
            label = np.asarray(label, np.float32)
            # image, label = transform(image, label)

            image = image.transpose((-1, 0, 1))
            for i in range(len(self.mean)):
                image[i, :, :] -= self.mean[i]
                image[i, :, :] /= self.std[i]

            return image.copy(), label.copy()
        else:
            img_file = self.img[index]
            name = img_file.split('\\')[-1]
            with h5py.File(img_file, 'r') as hf:
                image = hf['img'][:]
            image = np.asarray(image, np.float32)
            image = image.transpose((-1, 0, 1))
            for i in range(len(self.mean)):
                image[i, :, :] -= self.mean[i]
                image[i, :, :] /= self.std[i]
            return image.copy(),name

from PIL import Image
if __name__ == '__main__':

    train_dataset = LandslideDataSet(data_dir=r'E:\landslide_data\TrainData', set='labeled')
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, pin_memory=True)

    channels_sum, channel_squared_sum = 0, 0
    num_batches = len(train_loader)
    for data, label in train_loader:
        data = data.squeeze(0)[:3].permute(1, 2, 0).numpy()
        label = label.squeeze(0).numpy()
        plt.subplot(121)
        plt.imshow(data)
        plt.subplot(122)
        plt.imshow(label)
        plt.show()
        # break

# if __name__ == '__main__':
#
#     train_dataset = LandslideDataSet(data_dir=r'E:\landslide_data\TrainData',set='labeled')
#     train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, pin_memory=True)
#
#     channels_sum, channel_squared_sum = 0, 0
#     num_batches = len(train_loader)
#     # datas = []
#     # labels = []
#     for data, label in train_loader:
#         # datas.append(data)
#         # labels.append(label)
#         channels_sum += torch.mean(data, dim=[0, 2, 3])
#         channel_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
#
#     mean = channels_sum / num_batches
#     std = (channel_squared_sum / num_batches - mean ** 2) ** 0.5
#     print(mean, std)
#     # datas = np.array([i.numpy() for i in datas])
#     # datas = torch.FloatTensor(datas).squeeze(1)
#     # print(datas.shape)
#     # print(torch.mean(datas,dim=[0,2,3]))
#     # print(torch.std(datas,dim=[0, 2, 3]))
#     # [-0.4914, -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644, 0.0802, 0.3000, 0.4082, 0.0823, 0.0516, 0.3338, 0.7819]
#     # [0.9325, 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354, 0.8491, 0.9061, 1.6072, 0.8848, 0.9232, 0.9018, 1.2913]
