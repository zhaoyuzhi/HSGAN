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
        self.imglist = utils.get_pairs_name(opt.baseroot_train)
    
    def __getitem__(self, index):

        # read an image
        rgbpath = os.path.join(self.opt.baseroot_train, 'generated_rgb', self.imglist[index] + '.png')
        matpath = os.path.join(self.opt.baseroot_train, 'spectral', self.imglist[index] + '.mat')

        # read a rgb image
        rgb = cv2.imread(rgbpath)

        # read a spectral image
        spectral = io.loadmat(matpath)
        spectral = spectral['ref']
        
        # crop
        if self.opt.crop_size > 0:
            h, w = spectral.shape[:2]
            rand_h = random.randint(0, h - self.opt.crop_size)
            rand_w = random.randint(0, w - self.opt.crop_size)
            rgb = rgb[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]            # (256, 256, 31), in range [0, 1], float64
            spectral = spectral[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]  # (256, 256, 3), in range [0, 1], float64

        # normalization
        rgb = rgb / 255.0
        #spectral = spectral * 16.6

        # to tensor
        rgb = torch.from_numpy(rgb.astype(np.float32).transpose(2, 0, 1)).contiguous()
        spectral = torch.from_numpy(spectral.astype(np.float32).transpose(2, 0, 1)).contiguous()

        return rgb, spectral

    def __len__(self):
        return len(self.imglist)

class HS_multiscale_ValDSet(Dataset):
    def __init__(self, opt):
        self.opt = opt
        # build image list
        self.imglist = utils.get_pairs_name(opt.baseroot_val)
    
    def __getitem__(self, index):

        # read an image
        rgbpath = os.path.join(self.opt.baseroot_val, 'generated_rgb', self.imglist[index] + '.png')
        matpath = os.path.join(self.opt.baseroot_val, 'spectral', self.imglist[index] + '.mat')
        imgname = self.imglist[index]

        # read a rgb image
        rgb = cv2.imread(rgbpath)

        # read a spectral image
        spectral = io.loadmat(matpath)
        spectral = spectral['ref']
        
        '''
        # crop
        if self.opt.crop_size > 0:
            h, w = spectral.shape[:2]
            rand_h = random.randint(0, h - self.opt.crop_size)
            rand_w = random.randint(0, w - self.opt.crop_size)
            rgb = rgb[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]            # (256, 256, 31), in range [0, 1], float64
            spectral = spectral[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]  # (256, 256, 3), in range [0, 1], float64
        '''
        
        # normalization
        rgb = rgb / 255.0
        #spectral = spectral * 16.6

        # to tensor
        rgb = torch.from_numpy(rgb.astype(np.float32).transpose(2, 0, 1)).contiguous()
        spectral = torch.from_numpy(spectral.astype(np.float32).transpose(2, 0, 1)).contiguous()

        return rgb, spectral, imgname

    def __len__(self):
        return len(self.imglist)

if __name__ == "__main__":

    imgpath = 'E:\\dataset, task related\\Hyperspectral Imaging\\Harvard\\train\\spectral\\img3.mat'
    spectral = io.loadmat(imgpath)
    print(spectral)
    data = spectral['ref']
    print(data.shape)
    print(data.dtype)
    
    show = (data[:,:,20] * 16.6 * 255.0).astype(np.uint8) # 12bit data
    print(show)
    show = cv2.resize(show, (show.shape[1] // 4, show.shape[0] // 4))
    cv2.imshow('out', show)
    cv2.waitKey(0)

    calib = np.loadtxt('E:\\dataset, task related\\Hyperspectral Imaging\\Harvard\\CZ_hsdb\\calib.txt')
    print(calib.shape)
    print(calib.dtype)

    out = np.dot(data, calib)
    print(out.shape)
    out = (out * 255.0).astype(np.uint8)

    lbl = spectral['lbl']
    lbl = (lbl * 255.0).astype(np.uint8)

    show = np.concatenate((out, lbl), axis = 1)
    show = cv2.resize(show, (show.shape[1] // 4, show.shape[0] // 4))
    cv2.imshow('out', show)
    cv2.waitKey(0)
