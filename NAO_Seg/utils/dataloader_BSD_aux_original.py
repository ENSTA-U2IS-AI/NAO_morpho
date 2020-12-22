#ref: https://github.com/meteorshowers/RCF-pytorch
import torch
import os
import glob
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import random


def randomCrop(image, label):
    f_scale = 0.5 + random.randint(0, 11) / 10.0
    image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
    return image, label

class BSD_loader(Dataset):
    def __init__(self,root='./data/HED-BSDS',split='train',target_size=(256,256),transform=False,normalisation=False):
        # first: load imgs form indicated path
        self.root = root
        self.split = split
        self.transform = transform
        self.target_size=target_size
        self.normalisation=normalisation

        if self.split=='train':
            self.filelist = os.path.join(self.root, 'train_pair.lst')
        elif self.split == 'test':
            self.filelist = os.path.join(self.root, 'test.lst')

        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, item):
        if self.split=='train':
            img_file, lb_file = self.filelist[item].split()
            img=cv2.imread(os.path.join(self.root,img_file),cv2.IMREAD_COLOR).astype(np.float32)
            label= cv2.imread(os.path.join(self.root,lb_file),cv2.IMREAD_GRAYSCALE).astype(np.float32)


            img = cv2.resize(img, dsize=self.target_size, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, dsize=self.target_size, interpolation=cv2.INTER_LINEAR)

            label = label[np.newaxis, :, :]  # Add one channel at first (CHW).
            label[label==0] = 0
            label[np.logical_and(label>0, label<64)] = 2
            label[label >= 64] = 1
            label = label.astype(np.float32)    # To float32.
            # label = np.squeeze(label)

            if(self.normalisation==True):
                  img = img - np.array((104.00698793, # Minus statistics.
                                        116.66876762,
                                        122.67891434))
            else:
                  img = img

            img = np.transpose(img, (2, 0, 1))  # HWC to CHW.
            img = img.astype(np.float32)        # To float32.
            return img,label
        # else:
        #     img_file=self.filelist[item].rstrip()
        #     print(img_file)
        #     img_original = cv2.imread(os.path.join(self.root, img_file), cv2.IMREAD_COLOR).astype(np.float32)
        #     return img_original,img_file



if __name__=="__main__":
    root = str(os.getcwd().split('/utils')[0]) + "/data/HED-BSDS"
    bsd_dataset = BSD_loader(root=root,split='train')
    print(len(bsd_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=bsd_dataset,
                          batch_size = 2,
                          shuffle=True,pin_memory=True, num_workers=16)

    for i, (input, _) in enumerate(train_loader):
      # print('i:%d,img size:%s,label size:%s',i,input.size(),target.size())
      print(_.size())
      print(input.size())
      # print(input.max())
      # print(target.max())
      









