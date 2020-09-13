import torch
import os
import glob
from PIL import Image
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from torchvision.transforms.functional import to_grayscale,hflip
import matplotlib.pyplot as plt
import torch.nn as nn
from skimage.transform import resize
# from utils import mosaic

class BSD_loader(Dataset):
    def __init__(self,path,type='train',transform=None,target_size=(416,416)):
        # first: load imgs form indicated path
        self.path = path
        self.type = type
        self.imgs = glob.glob(os.path.join(path,'images',type,'*.jpg'))
        self.transform = transform
        self.target_size=target_size

    def __getitem__(self, item):
        img_path = self.imgs[item]
        label_path = img_path.replace('images','groundTruth')
        label_path = label_path.replace('jpg','bmp')
        img = Image.open(img_path).convert('RGB')
        img = img.copy()
        label = Image.open(label_path)
        label = label.copy()
       
        label=label.resize(self.target_size)
        img =img.resize(self.target_size)

        img = np.array(img)
        if img.max() > 1:
          img = img / 255
        img = img.astype(np.float32)
        if self.transform:
          img = self.transform(img)

        label = np.array(label)
        label[label>0]=1

        lable_transform = transforms.Compose([transforms.ToTensor()])
        label = lable_transform(label)
        label = torch.squeeze(label)
        # print(label)
        return img,label

    def __len__(self):
        return len(self.imgs)

if __name__=="__main__":
    data_path = str(os.getcwd().split('/utils')[0])+"/data/BSR/BSDS500/data/"
    print(data_path)
    bsd__dataset = BSD_loader(data_path,type='train',
                  transform = transforms.Compose([
                    transforms.ToTensor(),   # range [0, 255] -> [0.0,1.0]
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    #             std=[0.229, 0.224, 0.225])
                  ]))
    print(len(bsd__dataset))
    train_loader = torch.utils.data.DataLoader(dataset=bsd__dataset,
                          batch_size = 2,
                          shuffle=True,pin_memory=True, num_workers=16)
    # for img,label in train_loader:
    #   print(img.size())
    #   print(label.size())
    for i, (input, target) in enumerate(train_loader):
      # print('i:%d,img size:%s,label size:%s',i,input.size(),target.size())
      print(target.size())
      print(input.size())
      









