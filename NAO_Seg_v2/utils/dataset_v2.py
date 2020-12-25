import torch
import os
import glob
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import random

def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]

class BSD_loader(Dataset):
    def __init__(self,root='./data/HED-BSDS',split='train',target_size=(512,512),random_crop=False,random_flip=False,ignore_label=0,normalisation=False,keep_size=False):
        # first: load imgs form indicated path
        self.root = root
        self.type = type
        self.imgs_base = os.path.join(root,'images',split)
        self.target_size=target_size
        self.random_crop=random_crop
        self.random_flip=random_flip
        self.crop_h, self.crop_w = target_size
        self.ignore_label=ignore_label
        self.split = split
        self.normalisation=normalisation
        self.keep_size=keep_size

        if self.split=='train':
            self.filelist = os.path.join(self.root, 'train_pair.lst')
        else:
            self.imgs = glob.glob(os.path.join(root,'BSR/BSDS500/data/images',split,'*.jpg'))
        

    def randomCrop(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, item):
        if self.split=='train':
          img_file, lb_file = self.filelist[item].split()
          img=cv2.imread(os.path.join(self.root,img_file),cv2.IMREAD_COLOR).astype(np.float32)
          label= cv2.imread(os.path.join(self.root,lb_file),cv2.IMREAD_GRAYSCALE).astype(np.float32)
        else:
          img_path = self.imgs[item]
          label_path = img_path.replace('images','groundTruth')
          label_path = label_path.replace('jpg','bmp')
          img = cv2.imread(img_path,cv2.IMREAD_COLOR).astype(np.float32)
          label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

        #if random crop
        if(self.split=='train'):
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
            else:
                img = cv2.resize(img, dsize=self.target_size, interpolation=cv2.INTER_LINEAR)
                label = cv2.resize(label, dsize=self.target_size, interpolation=cv2.INTER_LINEAR)

        elif (self.split == 'val'):
            img = cv2.resize(img, dsize=self.target_size, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, dsize=self.target_size, interpolation=cv2.INTER_LINEAR)
        elif (self.split == 'test'):
            if(self.keep_size==True):
                if (img.shape[0] < img.shape[1]):
                    img = cv2.transpose(img)
                    label = cv2.transpose(label)
            else:
                img = cv2.resize(img, dsize=self.target_size, interpolation=cv2.INTER_LINEAR)
                label = cv2.resize(label, dsize=self.target_size, interpolation=cv2.INTER_LINEAR)
        # elif(self.split!='test'):
        #     img = cv2.resize(img, dsize=self.target_size, interpolation=cv2.INTER_LINEAR)
        #     label = cv2.resize(label, dsize=self.target_size, interpolation=cv2.INTER_NEAREST)
        # elif(self.split=='test'):
        #     if(img.shape[0]<img.shape[1]):
        #         img=cv2.transpose(img)
        #         label=cv2.transpose(label)


        if(self.random_flip):
            flip = np.random.choice(2) * 2 - 1
            img = img[:, :, ::flip]
            label = label[:, ::flip]

        if(self.normalisation==True):
            img = img - np.array((104.00698793, # Minus statistics.
                                  116.66876762,
                                  122.67891434))
        else:
            # img = img/255.
            img = img

        img = np.transpose(img, (2, 0, 1))  # HWC to CHW.
        img = img.astype(np.float32)        # To float32.

        label = label[np.newaxis, :, :]  # Add one channel at first (CHW).
        label[label < 127.5] = 0.0
        label[label >= 127.5] = 1.0
        label = label.astype(np.float32)
        label = np.squeeze(label)

        return img.copy(),label.copy()

    def __len__(self):
        if self.split=='train':
            return len(self.filelist)
        else:
            return len(self.imgs)

if __name__=="__main__":
    data_path = str(os.getcwd().split('/utils')[0])+"/data/BSR/BSDS500/data/"
    print(data_path)
    bsd__dataset = BSD_loader(root=data_path,split='train',random_crop=False,random_flip=False,normalisation=False)
    # print(len(bsd__dataset))
    train_loader = torch.utils.data.DataLoader(dataset=bsd__dataset,
                          batch_size = 1,
                          shuffle=True,pin_memory=True, num_workers=16)

    for i, (input, target) in enumerate(train_loader):
      # print('i:%d,img size:%s,label size:%s',i,input.size(),target.size())
      print(target.size())
      print(input.size())
      print(input.max())
      print(target.max())
      









