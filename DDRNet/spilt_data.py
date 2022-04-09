import shutil
import os
from sklearn.utils import shuffle
ori_img = r'E:\landslide_data\TrainData\img'
target_train = r'E:\landslide_data\train'
target_val = r'E:\landslide_data\val'

file_name = os.listdir(ori_img)
file_name = shuffle(file_name)

ratio = 0.98

for i in range(int(ratio*len(file_name))):
    img_ori = ori_img+'\\'+file_name[i]
    mask_ori = r'E:\landslide_data\TrainData\mask'+'\\'+file_name[i].replace('image','mask')
    shutil.copyfile(img_ori,target_train+'\\'+'img'+'\\'+file_name[i])
    shutil.copyfile(mask_ori, target_train + '\\' + 'mask' + '\\' + file_name[i].replace('image','mask'))

for i in range(int(ratio*len(file_name)),len(file_name)):
    img_ori = ori_img+'\\'+file_name[i]
    mask_ori = r'E:\landslide_data\TrainData\mask'+'\\'+file_name[i].replace('image','mask')
    shutil.copyfile(img_ori,target_val+'\\'+'img'+'\\'+file_name[i])
    shutil.copyfile(mask_ori, target_val + '\\' + 'mask' + '\\' + file_name[i].replace('image','mask'))
