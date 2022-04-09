import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from dataset import LandslideDataSet
from model_ import get_seg_model
from torch.utils import data
import h5py
from swin_transformer2 import SwinTransformer
from 尝试可视化h5文件 import get_f1


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# model_path = './logs/ep098-loss0.900-val_loss0.786.pth'
# model = get_seg_model(14)
# model.load_state_dict(torch.load(model_path))
# model.eval()
# dataset = data.DataLoader(LandslideDataSet(data_dir=r'E:\landslide_data\val', set='labeled'),batch_size=1)
# for img,label in dataset:
#     print(img.shape)
#     print(label.shape)
#     with torch.no_grad():
#         preds = model(img)
#
#     plt.subplot(221)
#     plt.imshow(label.squeeze(0).numpy())
#     plt.title('true')
#
#     _, pred = torch.max(preds[0], dim=1)
#     print(pred.shape)
#     plt.subplot(222)
#     plt.imshow(pred.squeeze(0).numpy())
#     plt.title('pred1')
#
#     plt.subplot(223)
#     _, pred = torch.max(preds[1], dim=1)
#     print(pred.shape)
#     plt.imshow(pred.squeeze(0).numpy())
#     plt.title('pred2')
#
#     plt.subplot(224)
#     _, pred0 = torch.max(preds[0], dim=1)
#     _, pred1 = torch.max(preds[1], dim=1)
#     pred = torch.logical_or(pred0,pred1)
#     print(pred.shape)
#     plt.imshow(pred.squeeze(0).numpy())
#     plt.title('pred2')
#     plt.show()

# dataset = data.DataLoader(LandslideDataSet(data_dir=r'E:\pycharm_project\compete\IARAI\DDRNet', set='unlabeled'),batch_size=1)
# for img ,name in dataset:
#     with torch.no_grad():
#         preds = model(img)
#     _,pred = torch.max(preds[0],dim=1)
#     pred = pred.squeeze(0).data.numpy().astype('uint8')
#
#     # _,pred = torch.max(pred,dim=1)
#     with h5py.File('./result/' + name[0].replace('image','mask'), 'w') as hf:
#         hf.create_dataset('mask', data=pred)
    # break

def read_pre(file):
    f = h5py.File(file, 'r')
    for group in f.keys():
        name = f[group].name
        shape = f[group].shape
        value = np.array(f[group][:])
    return name, shape, value


def read_train(file):
    f = h5py.File(file, 'r')
    for group in f.keys():
        # print(f[group])
        name = f[group].name
        shape = f[group].shape
        value = np.array(f[group][:])
        # print(name,shape,value)
    return name, shape, value

def get_result(model,dataset):
    for img ,name in dataset:
        img = img.cuda()
        with torch.no_grad():
            preds = model(img)
        _, pred1 = torch.max(preds[0], dim=1)
        _, pred2 = torch.max(preds[1], dim=1)
        pred = torch.logical_or(pred1, pred2)
        pred = pred.squeeze(0).data.cpu().numpy().astype('uint8')

        # _,pred = torch.max(pred,dim=1)
        with h5py.File('E:/landslide_data/val/result/' + name[0].replace('image','mask'), 'w') as hf:
            hf.create_dataset('mask', data=pred)

def get_result_trans(model,dataset):
    for img ,name in dataset:
        img = img.cuda()
        with torch.no_grad():
            pred = model(img)
        _, pred = torch.max(pred, dim=1)
        pred = pred.squeeze(0).data.cpu().numpy().astype('uint8')
        with h5py.File('E:/landslide_data/val/result/' + name[0].replace('image','mask'), 'w') as hf:
            hf.create_dataset('mask', data=pred)

def get_result_trans_pre(model,dataset):
    for img ,name in dataset:
        img = img.cuda()
        with torch.no_grad():
            pred = model(img)
        _, pred = torch.max(pred, dim=1)
        pred = pred.squeeze(0).data.cpu().numpy().astype('uint8')
        with h5py.File('E:/pycharm_project/compete/IARAI/DDRNet/submission_name/' + name[0].replace('image','mask'), 'w') as hf:
            hf.create_dataset('mask', data=pred)

def get_result1(model,dataset):
    for img ,name in dataset:
        img = img.cuda()
        with torch.no_grad():
            preds = model(img)
        _, pred = torch.max(preds[0], dim=1)
        pred = pred.squeeze(0).data.cpu().numpy().astype('uint8')

        # _,pred = torch.max(pred,dim=1)
        with h5py.File('E:/landslide_data/val/result/' + name[0].replace('image','mask'), 'w') as hf:
            hf.create_dataset('mask', data=pred)

def get_result_(model,dataset):
    for img ,name in dataset:
        img = img
        with torch.no_grad():
            preds = model(img)
        _,pred1 = torch.max(preds[0],dim=1)
        _, pred2 = torch.max(preds[1], dim=1)
        pred = torch.logical_or(pred1,pred2)
        pred = pred.squeeze(0).data.numpy().astype('uint8')

        # _,pred = torch.max(pred,dim=1)
        with h5py.File('E:/pycharm_project/compete/IARAI/DDRNet/submission_name/' + name[0].replace('image','mask'), 'w') as hf:
            hf.create_dataset('mask', data=pred)
        # with h5py.File('E:/landslide_data/val/result/' + name[0].replace('image','mask'), 'w') as hf:
        #     hf.create_dataset('mask', data=pred)

if __name__ == '__main__':
    mean = [0.9257, 0.9227, 0.9541, 0.9596, 1.0228, 1.0426, 1.0358, 1.0468, 1.1699,
        1.1736, 1.0495, 1.0370, 1.2511, 1.6495]
    std = [0.1410, 0.2207, 0.3184, 0.5724, 0.4601, 0.4465, 0.4651, 0.4948, 0.5133,
        0.6836, 0.5323, 0.6628, 0.6784, 1.0727]
        
    model_path = './logs/sub1.pth'
    model = get_seg_model(14)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    while True:
        img_name = input('please input image name:')
        name, shape, image = read_pre(img_name)
       
        plt.subplot(121)
        plt.imshow(image[...,1:4])
        plt.axis('off')
        plt.title('origin')
        
        plt.subplot(122)
        image = image.transpose((-1, 0, 1))
        for i in range(len(mean)):
            image[i, :, :] -= mean[i]
            image[i, :, :] /= std[i]
        image = torch.FloatTensor(image).unsqueeze(0)
        image =  model(image)[0].squeeze(0)
        _,image = torch.max(image,dim=0)
        plt.imshow(image)
        plt.axis('off')
        plt.title('predict')
        
        plt.show()





# import numpy as np
# if __name__ == '__main__':
#     def read_pre(file):
#         f = h5py.File(file, 'r')
#         for group in f.keys():
#             name = f[group].name
#             shape = f[group].shape
#             value = np.array(f[group][:])
#         return name, shape, value
#
#
#     def read_train(file):
#         f = h5py.File(file, 'r')
#         for group in f.keys():
#             # print(f[group])
#             name = f[group].name
#             shape = f[group].shape
#             value = np.array(f[group][:])
#             # print(name,shape,value)
#         return name, shape, value
#
#     # for i in os.listdir(r'E:\landslide_data\val\mask'):
#     #     name, shape, value1 = read_train(r'E:\landslide_data\val\mask\%s'%i)
#     #     name, shape, value2 = read_pre(r'E:\landslide_data\val\result\%s'%i)
#     #     plt.subplot(121)
#     #     plt.imshow(value1)
#     #     plt.axis('off')
#     #     plt.title('true')
#     #
#     #     plt.subplot(122)
#     #     plt.imshow(value2)
#     #     plt.axis('off')
#     #     plt.title('pred')
#     #     plt.show()
#     f1 = get_f1(r'E:\landslide_data\val\result', r'E:\landslide_data\val\mask')
#     print(f1)