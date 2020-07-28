import torch
import os
import glob
import cv2
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class BSD_loader(Dataset):
    def __init__(self,path,type='train'):
        # first: load imgs form indicated path
        self.path = path
        self.type = type
        self.imgs = glob.glob(os.path.join(path,'images',type,'*.jpg'))

    def __getitem__(self, item):
        img_path = self.imgs[item]
        label_path = img_path.replace('images','groundTruth')
        label_path = label_path.replace('jpg','bmp')
        img = cv2.imread(img_path)
        img = img.copy()
        label = cv2.imread(label_path)
        label = label.copy()
     
        if img.shape[0]>img.shape[1]:
          img = np.rot90(img)
          label = np.rot90(label)
          # img = img.transpose(Image.ROTATE_90)
          # label = label.transpose(Image.ROTATE_90)
        img = img.reshape(3, img.shape[0], img.shape[1])
        label = label.reshape(3, label.shape[0], label.shape[1])
        if img.max()>1:
          img = img/255
        if label.max()>1:
          label = label/255
        # print(label)
        return img,label
    def __len__(self):
        return len(self.imgs)

if __name__=="__main__":
    data_path = str(os.getcwd().split('/utils')[0])+"/data/BSR/BSDS500/data/"
    print(data_path)
    bsd__dataset = BSD_loader(data_path,type='train')
    print(len(bsd__dataset))
    train_loader = torch.utils.data.DataLoader(dataset=bsd__dataset,
                          batch_size = 2,
                          shuffle=True)
    for img,label in train_loader:
      print(label.shape)










