import torch
import os
import glob
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import random

class BSD_loader(Dataset):
    def __init__(self,path,type='train',transform=None,target_size=(416,416),random_crop=False,random_flip=False,ignore_label=0):
        # first: load imgs form indicated path
        self.path = path
        self.type = type
        self.imgs = glob.glob(os.path.join(path,'images',type,'*.jpg'))
        self.transform = transform
        self.target_size=target_size
        self.random_crop=random_crop
        self.random_flip=random_flip
        self.crop_h, self.crop_w = target_size
        self.ignore_label=ignore_label

    def randomCrop(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, item):
        img_path = self.imgs[item]
        label_path = img_path.replace('images','groundTruth')
        label_path = label_path.replace('jpg','bmp')
        img = cv2.imread(img_path,cv2.IMREAD_COLOR).astype(np.float32)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

        #if random crop
        if(self.random_crop):
            img,label=self.randomCrop(img,label)

        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(img, 0, pad_h, 0,
                                         pad_w, cv2.BORDER_CONSTANT,
                                         value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=(self.ignore_label,))
        else:
            img_pad, label_pad = img, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        img = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)


        if(self.random_flip):
            flip = np.random.choice(2) * 2 - 1
            img = img[:, :, ::flip]
            label = label[:, ::flip]

        img = img - np.array((104.00698793, # Minus statistics.
                              116.66876762,
                              122.67891434))
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW.
        img = img.astype(np.float32)        # To float32.

        label = label[np.newaxis, :, :]  # Add one channel at first (CHW).
        label[label < 127.5] = 0.0
        label[label >= 127.5] = 1.0
        label = label.astype(np.float32)

        return img.copy(),label.copy()

        # img = Image.open(img_path).convert('RGB')
        # img = img.copy()
        # label = Image.open(label_path)
        # label = label.copy()
        #
        # label=label.resize(self.target_size)
        # img =img.resize(self.target_size)
        #
        # img = np.array(img)
        # if img.max() > 1:
        #     img = img-np.array((104.00698793,  # Minus statistics.
        #                         116.66876762,
        #                         122.67891434))
        #     img = img / 255
        # img = img.astype(np.float32)
        # if self.transform:
        #   img = self.transform(img)
        #
        # label = np.array(label)
        # label[label>0]=1
        #
        # lable_transform = transforms.Compose([transforms.ToTensor()])
        # label = lable_transform(label)
        # label = torch.squeeze(label)
        # # print(label)
        # return img,label

    def __len__(self):
        return len(self.imgs)

if __name__=="__main__":
    data_path = str(os.getcwd().split('/utils')[0])+"/data/BSR/BSDS500/data/"
    print(data_path)
    bsd__dataset = BSD_loader(data_path,type='train',random_crop=True,random_flip=True)
    print(len(bsd__dataset))
    train_loader = torch.utils.data.DataLoader(dataset=bsd__dataset,
                          batch_size = 2,
                          shuffle=True,pin_memory=True, num_workers=16)

    for i, (input, target) in enumerate(train_loader):
      # print('i:%d,img size:%s,label size:%s',i,input.size(),target.size())
      print(target.size())
      print(input.size())
      print(input.max())
      print(target.max())
      









