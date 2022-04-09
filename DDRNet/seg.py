from swin_transformer2 import SwinTransformer
import torch
from torch.utils.data import DataLoader,distributed,BatchSampler
import torch.nn as nn
from tqdm import tqdm
import einops
import torch.distributed as td
import os
import numpy as np
from loss import FocalLoss
from fit import fit_epoch,fit_epoch_trans
import os
import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils import data
import torch.backends.cudnn as cudnn
from dataset import LandslideDataSet,LandslideDataSet_aug
from loss import OhemCrossEntropy,FocalLoss
from model_ import get_seg_model




def transfer_model(pretrained_file, model):
    pretrained_dict = torch.load(pretrained_file)  # get pretrained dict
    model_dict = model.state_dict()  # get model dict
    print('模型结构数：', len(model_dict))
    # 在合并前(update),需要去除pretrained_dict一些不需要的参数
    pretrained_dict = transfer_state_dict(pretrained_dict, model_dict)
    print('-------------------------------------')
    # transfer_state_dict(model_dict, pretrained_dict)
    print('预训练结构数：', len(pretrained_dict))
    model_dict.update(pretrained_dict)  # 更新(合并)模型的参数
    model.load_state_dict(model_dict)
    return model


def transfer_state_dict(pretrained_dict, model_dict):
    # state_dict2 = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    state_dict = {}
    # for k1,k2 in zip(pretrained_dict.keys(),model_dict.keys()):
    #     print(k1,k2)
    # for k, v in pretrained_dict['model'].items():
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(v) == np.shape(model_dict[k]):
            # state_dict.setdefault(k, v)
            state_dict[k] = v
        else:
            print("Missing key(s) in state_dict :{}".format(k))


    return state_dict

def train2():
    cudnn.enabled = True
    cudnn.benchmark = True
    model = SwinTransformer(img_size=128, patch_size=4, in_chans=14, num_classes=2,
                            embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=4,
                            mlp_ratio=4., drop_path_rate=0.2)
    # model.init_pretrained('DDRNet23s_imagenet.pth')

    # model = transfer_model('DDRNet23s_imagenet.pth',model)
    # model = transfer_model('./swin_tiny_patch4_window7_224.pth', model)
    model = transfer_model('./logs/ep036-loss0.073-val_loss0.102_f10.706.pth',model)
    # pt = './swin_tiny_patch4_window7_224.pth'
    # pretrain_para = torch.load(pt)
    # msd = {}
    # for k, v in pretrain_para.items():
    #     msd.update({k.replace('module.', ''): v})
    # print('预训练长度:',len(msd))
    # model.load_state_dict(msd)
    model.train()
    model = model.cuda()

    cross_entropy_loss = OhemCrossEntropy()
    focalloss = FocalLoss(2,alpha=torch.tensor([1,0.8]))#.to('cuda:{}'.format(rank))
    # src_loader = data.DataLoader(
    #     LandslideDataSet(data_dir=r'E:\landslide_data\train', set='labeled'),
    #     batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = data.DataLoader(
        LandslideDataSet(data_dir=r'E:\landslide_data\val', set='labeled'),
        batch_size=38, shuffle=False, num_workers=4, pin_memory=True)
    dataset = data.DataLoader(LandslideDataSet(data_dir=r'E:\landslide_data\val', set='unlabeled'),
                              batch_size=1)

    src_loader = data.DataLoader(
        LandslideDataSet(data_dir=r'E:\landslide_data\train', set='labeled'),
        batch_size=48, shuffle=True, num_workers=8, pin_memory=True)

    src_aug_loader = data.DataLoader(
        LandslideDataSet_aug(data_dir=r'E:\landslide_data\train', set='labeled'),
        batch_size=48, shuffle=True, num_workers=8, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4,
                                 amsgrad=False)
    T1 = 29  # 波动周期
    T2 = 20  # 半周期
    lr_lambda = lambda x: ((np.cos(2 * np.pi * x / T1) + 1) * (1 - 1E-3) / 2 + 1e-3) * np.exp(-x / T2 * 0.693147)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for name, param in model.named_parameters():
        param.requires_grad = True
    for i in range(50):
        fit_epoch_trans(src_aug_loader, test_loader, dataset, model, optimizer, cross_entropy_loss, focalloss, lr_scheduler, i,
                  Epoch=50)
    for i in range(50):
        fit_epoch_trans(src_loader, test_loader, dataset, model, optimizer, cross_entropy_loss, focalloss, lr_scheduler, i+50,
                  Epoch=50)

if __name__ == '__main__':
    train2()