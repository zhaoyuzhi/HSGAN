import os
import random
import numpy as np
import cv2
import scipy.io as io
import torch
from torch.utils.data import Dataset

import dataset.utils as utils

class HS_multiscale_DSet(Dataset):
    def __init__(self, opt):
        self.opt = opt
        # build image list
        self.imglist = utils.get_bmps(opt.baseroot_train)
    
    def __getitem__(self, index):

        # read an image
        imgpath = self.imglist[index]
        img = cv2.imread(imgpath)

        # read a spectral image
        object_name = imgpath.split('/')[-2]
        folder_name = imgpath[ : (len(imgpath) - len(imgpath.split('/')[-1]))]
        spectral_img = np.zeros([512, 512, 31], dtype = np.uint8)
        for i in range(31):
            n = str(i + 1)
            n = n.zfill(2)
            s = object_name + '_' + n + '.png'
            s = os.path.join(folder_name, s)
            simg = cv2.imread(s)[:, :, 0]
            spectral_img[:, :, i] = simg

        # crop
        if self.opt.crop_size > 0:
            h, w = img.shape[:2]
            rand_h = random.randint(0, h - self.opt.crop_size)
            rand_w = random.randint(0, w - self.opt.crop_size)
            img = img[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]                        # (256, 256, 31), in range [0, 1], float64
            spectral_img = spectral_img[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]      # (256, 256, 3), in range [0, 1], float64

        # normalization
        img = img.astype(np.float64) / 255.0
        spectral_img = spectral_img.astype(np.float64) / 255.0

        # to tensor
        img = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1)).contiguous()
        spectral_img = torch.from_numpy(spectral_img.astype(np.float32).transpose(2, 0, 1)).contiguous()

        return img, spectral_img

    def __len__(self):
        return len(self.imglist)

class HS_multiscale_ValDSet(Dataset):
    def __init__(self, opt):
        self.opt = opt
        # build image list
        self.imglist = utils.get_bmps(opt.baseroot_val)
    
    def __getitem__(self, index):

        # read an image
        imgpath = self.imglist[index]
        img = cv2.imread(imgpath)
        imgname = imgpath.split('\\')[-1]

        # read a spectral image
        object_name = imgpath.split('\\')[-2]
        folder_name = imgpath[ : (len(imgpath) - len(imgpath.split('\\')[-1]))]
        spectral_img = np.zeros([512, 512, 31], dtype = np.uint8)
        for i in range(31):
            n = str(i + 1)
            n = n.zfill(2)
            s = object_name + '_' + n + '.png'
            s = os.path.join(folder_name, s)
            simg = cv2.imread(s)[:, :, 0]
            spectral_img[:, :, i] = simg

        '''
        # crop
        if self.opt.crop_size > 0:
            h, w = img.shape[:2]
            rand_h = random.randint(0, h - self.opt.crop_size)
            rand_w = random.randint(0, w - self.opt.crop_size)
            img = img[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]                        # (256, 256, 31), in range [0, 1], float64
            spectral_img = spectral_img[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]      # (256, 256, 3), in range [0, 1], float64
        '''

        # normalization
        img = img.astype(np.float64) / 255.0
        spectral_img = spectral_img.astype(np.float64) / 255.0

        # to tensor
        img = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1)).contiguous()
        spectral_img = torch.from_numpy(spectral_img.astype(np.float32).transpose(2, 0, 1)).contiguous()

        return img, spectral_img, imgname

    def __len__(self):
        return len(self.imglist)

if __name__ == "__main__":

    import utils
    import cv2
    import os

    path = 'F:\\dataset, task related\\Hyperspectral Imaging\\CAVE'
    ret = utils.get_bmps(path)
    ret = ret[0]
    print(ret)
    object_name = ret.split('\\')[-2]
    folder_name = ret[ : (len(ret) - len(ret.split('\\')[-1]))]
    print(object_name)
    print(folder_name)
    spectral_img = np.zeros([512, 512, 31], dtype = np.uint8)
    for i in range(31):
        n = str(i + 1)
        n = n.zfill(2)
        s = object_name + '_' + n + '.png'
        s = os.path.join(folder_name, s)
        simg = cv2.imread(s)[:, :, 0]
        spectral_img[:, :, i] = simg
    show = spectral_img[:, :, 10]
    cv2.imshow('show', show)
    cv2.waitKey(0)
    '''
    imgpath = 'F:\\dataset, task related\\Hyperspectral Imaging\\CAVE\\val\\balloons_ms\\balloons_RGB.bmp'
    img = cv2.imread(imgpath)
    print(img.dtype)
    print(img.shape)
    print(imgpath[-3:])
    imgpath = 'F:\\dataset, task related\\Hyperspectral Imaging\\CAVE\\val\\balloons_ms\\balloons_ms_11.png'
    img = cv2.imread(imgpath)
    print(img.dtype)
    print(img.shape)
    show = img[:, :, 0]
    cv2.imshow('show', show)
    cv2.waitKey(0)
    '''
