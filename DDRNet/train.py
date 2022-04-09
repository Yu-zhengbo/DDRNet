from fit import fit_epoch
import os
import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils import data
import torch.backends.cudnn as cudnn
from dataset import LandslideDataSet
from loss import OhemCrossEntropy,FocalLoss
from model_ import get_seg_model
name_classes = ['Non-Landslide', 'Landslide']
epsilon = 1e-14
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


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
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(v) == np.shape(model_dict[k]):
            # state_dict.setdefault(k, v)
            state_dict[k] = v
        else:
            print("Missing key(s) in state_dict :{}".format(k))


    return state_dict


def main():
    cudnn.enabled = True
    cudnn.benchmark = True
    model = get_seg_model(inchanel=14, num_classes=2)
    model.train()
    model = model.cuda()

    cross_entropy_loss = OhemCrossEntropy()

    src_loader = data.DataLoader(
        LandslideDataSet(data_dir=r'E:\landslide_data\train', set='labeled'),
        batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = data.DataLoader(
        LandslideDataSet(data_dir=r'E:\landslide_data\val', set='labeled'),
        batch_size=38, shuffle=False, num_workers=4, pin_memory=True)



    optimizer = optim.AdamW(model.parameters(),
                           lr=1e-3, weight_decay=1e-2)
    if False:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

    num_epoch = len(src_loader)
    num_epoch_val = len(test_loader)
    for i in range(100):
        print('Start train')
        train_loss = 0
        val_loss = 0
        model.train()
        with tqdm(total=num_epoch, desc=f'Epoch {i + 1}/{100}', postfix=dict, mininterval=0.3) as pbar:
            for batch_id, src_data in enumerate(src_loader):
                if batch_id == num_epoch:
                    break
                optimizer.zero_grad()

                images, labels = src_data
                images = images.cuda()
                pred_interp = model(images)


                # CE Loss
                labels = labels.cuda().long()
                cross_entropy_loss_value = cross_entropy_loss(pred_interp, labels)
                cross_entropy_loss_value.backward()
                optimizer.step()
                train_loss += cross_entropy_loss_value.item()
                pbar.set_postfix(**{'loss': train_loss / (batch_id + 1),
                                    'lr': optimizer.state_dict()['param_groups'][0]['lr']})
                pbar.update(1)
        print('Start val')
        model.eval()
        with tqdm(total=num_epoch_val, desc=f'Epoch {i + 1}/{100}', postfix=dict, mininterval=0.3) as pbar:
            for batch_id, src_data in enumerate(test_loader):
                if batch_id == num_epoch_val:
                    break
                images, labels = src_data
                images = images.cuda()
                pred_interp = model(images)

                # CE Loss
                labels = labels.cuda().long()
                cross_entropy_loss_value = cross_entropy_loss(pred_interp, labels)
                val_loss += cross_entropy_loss_value.item()
                pbar.set_postfix(**{'loss': val_loss / (batch_id + 1),
                                    'lr': optimizer.state_dict()['param_groups'][0]['lr']})
                pbar.update(1)
        lr_scheduler.step()
        torch.save(model.state_dict(),
                   'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, train_loss / num_epoch, val_loss / num_epoch_val))
        print('Epoch end!')

def main():
    cudnn.enabled = True
    cudnn.benchmark = True
    model = get_seg_model(1)
    # model.init_pretrained('DDRNet23s_imagenet.pth')

    #model = transfer_model('DDRNet23s_imagenet.pth',model)
    model = transfer_model('DDRNet39_imagenet.pth',model)


    model.train()
    model = model.cuda()

    cross_entropy_loss = OhemCrossEntropy()
    focalloss = FocalLoss(2, alpha=torch.tensor([0.4, 0.4]))
    src_loader = data.DataLoader(
        LandslideDataSet(data_dir=r'E:\landslide_data\train', set='labeled'),
        batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = data.DataLoader(
        LandslideDataSet(data_dir=r'E:\landslide_data\val', set='labeled'),
        batch_size=38, shuffle=False, num_workers=4, pin_memory=True)
    dataset = data.DataLoader(LandslideDataSet(data_dir=r'E:\landslide_data\val', set='unlabeled'),
                              batch_size=1)
    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)

    optimizer = optim.AdamW(g0,lr=1e-3, weight_decay=1e-3)
    optimizer.add_param_group({'params': g1, 'weight_decay': 1e-3})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2})  # add g2 (biases)
    del g0, g1, g2

    if False:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)
    for name, param in model.named_parameters():
            param.requires_grad = False
    for name, param in model.named_parameters():
        if 'spp' in name or 'seghead_extra' in name or 'final_layer' in name:
            param.requires_grad = True
    # for i in range(100):
    #     fit_epoch(src_loader, test_loader,dataset, model, optimizer, cross_entropy_loss,focalloss, lr_scheduler, i, Epoch=100)


    model = transfer_model('logs/ep100-loss1.507-val_loss1.507_f10.001.pth', model)
    src_loader = data.DataLoader(
        LandslideDataSet(data_dir=r'E:\landslide_data\train', set='labeled'),
        batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)
    optimizer = optim.AdamW(g0, lr=1e-4, weight_decay=1e-3)
    optimizer.add_param_group({'params': g1, 'weight_decay': 1e-3})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2})  # add g2 (biases)
    del g0, g1, g2
    if False:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)
    for name, param in model.named_parameters():
            param.requires_grad = True
    for i in range(100):
        fit_epoch(src_loader, test_loader,dataset, model, optimizer, cross_entropy_loss,focalloss, lr_scheduler, i+100, Epoch=200)

    #     if 'conv1' not in name or 'last_layer' not in name:

def main():
    cudnn.enabled = True
    cudnn.benchmark = True
    model = get_seg_model(1)
    # model.init_pretrained('DDRNet23s_imagenet.pth')

    #model = transfer_model('DDRNet23s_imagenet.pth',model)
    model = transfer_model('./logs/sub1.pth',model)


    model.train()
    model = model.cuda()

    cross_entropy_loss = OhemCrossEntropy()
    focalloss = FocalLoss(2, alpha=torch.tensor([0.4, 0.4]))
    src_loader = data.DataLoader(
        LandslideDataSet(data_dir=r'E:\landslide_data\train', set='labeled'),
        batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = data.DataLoader(
        LandslideDataSet(data_dir=r'E:\landslide_data\val', set='labeled'),
        batch_size=38, shuffle=False, num_workers=4, pin_memory=True)
    dataset = data.DataLoader(LandslideDataSet(data_dir=r'E:\landslide_data\val', set='unlabeled'),
                              batch_size=1)
    # g0, g1, g2 = [], [], []  # optimizer parameter groups
    # for v in model.modules():
    #     if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
    #         g2.append(v.bias)
    #     if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
    #         g0.append(v.weight)
    #     elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
    #         g1.append(v.weight)
    #
    # optimizer = optim.AdamW(g0,lr=1e-3, weight_decay=1e-3)
    # optimizer.add_param_group({'params': g1, 'weight_decay': 1e-3})  # add g1 with weight_decay
    # optimizer.add_param_group({'params': g2})  # add g2 (biases)
    # del g0, g1, g2
    #
    # if False:
    #     lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
    # else:
    #     lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)
    # for name, param in model.named_parameters():
    #         param.requires_grad = False
    # for name, param in model.named_parameters():
    #     if 'spp' in name or 'seghead_extra' in name or 'final_layer' in name:
    #         param.requires_grad = True
    # for i in range(100):
    #     fit_epoch(src_loader, test_loader,dataset, model, optimizer, cross_entropy_loss,focalloss, lr_scheduler, i, Epoch=100)


    # model = transfer_model('logs/ep100-loss1.507-val_loss1.507_f10.001.pth', model)
    src_loader = data.DataLoader(
        LandslideDataSet(data_dir=r'E:\landslide_data\train', set='labeled'),
        batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)
    optimizer = optim.AdamW(g0, lr=1e-2, weight_decay=1e-3)
    optimizer.add_param_group({'params': g1, 'weight_decay': 1e-3})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2})  # add g2 (biases)
    del g0, g1, g2
    if False:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)
    else:

        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    for name, param in model.named_parameters():
            param.requires_grad = True
    for i in range(50):
        fit_epoch(src_loader, test_loader,dataset, model, optimizer, cross_entropy_loss,focalloss, lr_scheduler, i, Epoch=50)

    #     if 'conv1' not in name or 'last_layer' not in name:

if __name__ == '__main__':
    main()
