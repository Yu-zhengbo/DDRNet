import torch
from tqdm import tqdm
from 尝试可视化h5文件 import get_f1
from predict import get_result,get_result_,get_result_trans
def fit_epoch(src_loader,test_loader,dataset,model,optimizer,cross_entropy_loss,focalLoss,lr_scheduler,epoch,Epoch=100):
    num_epoch = len(src_loader)
    num_epoch_val = len(test_loader)
    print('Start Epoch:')
    train_loss = 0
    val_loss = 0
    model.train()
    with tqdm(total=num_epoch, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
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
            # focal_loss1 = focalLoss(pred_interp[0],labels)
            # focal_loss2 = focalLoss(pred_interp[1], labels)
            # print(cross_entropy_loss_value.item(),focal_loss1.item(),focal_loss2.item())
            # loss_epoch = cross_entropy_loss_value+focal_loss1+focal_loss2
            loss_epoch = cross_entropy_loss_value
            loss_epoch.backward()
            optimizer.step()
            train_loss += loss_epoch.item()
            pbar.set_postfix(**{'loss': train_loss / (batch_id + 1),
                                'lr': optimizer.state_dict()['param_groups'][0]['lr']})
            pbar.update(1)
    model.eval()
    with tqdm(total=num_epoch_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for batch_id, src_data in enumerate(test_loader):
            if batch_id == num_epoch_val:
                break
            images, labels = src_data
            images = images.cuda()
            pred_interp = model(images)

            # CE Loss
            labels = labels.cuda().long()
            cross_entropy_loss_value = cross_entropy_loss(pred_interp, labels)
            # focal_loss1 = focalLoss(pred_interp[0], labels)
            # focal_loss2 = focalLoss(pred_interp[1], labels)
            # loss_epoch = cross_entropy_loss_value + focal_loss1 + focal_loss2
            loss_epoch = cross_entropy_loss_value
            val_loss += loss_epoch.item()
            pbar.set_postfix(**{'loss': val_loss / (batch_id + 1),
                                'lr': optimizer.state_dict()['param_groups'][0]['lr']})
            pbar.update(1)
    lr_scheduler.step()
    get_result(model,dataset=dataset)
    f1 = get_f1(r'E:\landslide_data\val\result', r'E:\landslide_data\val\mask')
    print(f1)
    torch.save(model.state_dict(),
               'logs/ep%03d-loss%.3f-val_loss%.3f_f1%.3f.pth' % (
               epoch + 1, train_loss / num_epoch, val_loss / num_epoch_val,f1))
    print('Epoch end!')


def fit_epoch_trans(src_loader,test_loader,dataset,model,optimizer,cross_entropy_loss,focalLoss,lr_scheduler,epoch,Epoch=100):
    num_epoch = len(src_loader)
    num_epoch_val = len(test_loader)
    print('Start Epoch:')
    train_loss = 0
    val_loss = 0
    model.train()
    with tqdm(total=num_epoch, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for batch_id, src_data in enumerate(src_loader):
            if batch_id == num_epoch:
                break
            optimizer.zero_grad()

            images, labels = src_data
            images = images.cuda()
            pred_interp = model(images)

            # CE Loss
            labels = labels.cuda().long()
            cross_entropy_loss_value = focalLoss(pred_interp, labels)
            # focal_loss1 = focalLoss(pred_interp[0],labels)
            # focal_loss2 = focalLoss(pred_interp[1], labels)
            # print(cross_entropy_loss_value.item(),focal_loss1.item(),focal_loss2.item())
            # loss_epoch = cross_entropy_loss_value+focal_loss1+focal_loss2
            loss_epoch = cross_entropy_loss_value
            loss_epoch.backward()
            optimizer.step()
            train_loss += loss_epoch.item()
            pbar.set_postfix(**{'loss': train_loss / (batch_id + 1),
                                'lr': optimizer.state_dict()['param_groups'][0]['lr']})
            pbar.update(1)
    model.eval()
    with tqdm(total=num_epoch_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for batch_id, src_data in enumerate(test_loader):
            if batch_id == num_epoch_val:
                break
            images, labels = src_data
            images = images.cuda()
            pred_interp = model(images)

            # CE Loss
            labels = labels.cuda().long()
            cross_entropy_loss_value = focalLoss(pred_interp, labels)
            # focal_loss1 = focalLoss(pred_interp[0], labels)
            # focal_loss2 = focalLoss(pred_interp[1], labels)
            # loss_epoch = cross_entropy_loss_value + focal_loss1 + focal_loss2
            loss_epoch = cross_entropy_loss_value
            val_loss += loss_epoch.item()
            pbar.set_postfix(**{'loss': val_loss / (batch_id + 1),
                                'lr': optimizer.state_dict()['param_groups'][0]['lr']})
            pbar.update(1)
    lr_scheduler.step()
    get_result_trans(model,dataset=dataset)
    f1 = get_f1(r'E:\landslide_data\val\result', r'E:\landslide_data\val\mask')
    print(f1)
    torch.save(model.state_dict(),
               'logs/ep%03d-loss%.3f-val_loss%.3f_f1%.3f.pth' % (
               epoch + 1, train_loss / num_epoch, val_loss / num_epoch_val,f1))
    print('Epoch end!')