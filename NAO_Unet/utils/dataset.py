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

class BSD_loader(Dataset):
    def __init__(self,path,type='train',transform=None):
        # first: load imgs form indicated path
        self.path = path
        self.type = type
        self.imgs = glob.glob(os.path.join(path,'images',type,'*.jpg'))
        self.transform = transform

    def __getitem__(self, item):
        img_path = self.imgs[item]
        label_path = img_path.replace('images','groundTruth')
        label_path = label_path.replace('jpg','bmp')
        img = Image.open(img_path)
        img = img.copy()
        label = Image.open(label_path)
        label = label.copy()
        
        # here we don't use hfilp, because this fonction don't change the size.
        if img.size[0]<img.size[1]:
          img = img.transpose(Image.ROTATE_90)
          label = label.transpose(Image.ROTATE_90)
        
        if self.transform:
          img = self.transform(img)

        label = to_grayscale(label)
        # if label.max() > 1:
        #     label = label / 255
            
        lable_transform = transforms.Compose([transforms.ToTensor()])
        label = lable_transform(label)
        label = torch.squeeze(label)
        return img,label

    def __len__(self):
        return len(self.imgs)

if __name__=="__main__":
    data_path = str(os.getcwd().split('/utils')[0])+"/data/BSR/BSDS500/data/"
    print(data_path)
    bsd__dataset = BSD_loader(data_path,type='train',
                  transform = transforms.Compose([
                    transforms.ToTensor(),   # range [0, 255] -> [0.0,1.0]
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                  ]))
    print(len(bsd__dataset))
    train_loader = torch.utils.data.DataLoader(dataset=bsd__dataset,
                          batch_size = 2,
                          shuffle=True,pin_memory=True, num_workers=16)
    # for img,label in train_loader:
    #   print(img.size())
    #   print(label.size())
    for i, (input, target) in enumerate(train_loader):
      print('i:%d,img size:%s,label size:%s',i,input.size(),target.size())
      # print(target.cpu().numpy().astype('int64'))
      print(torch.max(input,1)[1].numpy()[0::].shape)
    # m = nn.Dropout(p=0.2)
    # input = torch.randn(20, 16)
    # output = m(input)
    # print(input)
    # print(output)









