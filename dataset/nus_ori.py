import os
import random
import numpy as np
import cv2
import scipy.io as io
import h5py
import torch
from torch.utils.data import Dataset

import dataset.utils as utils

class HS_multiscale_DSet(Dataset):
    def __init__(self, opt):
        self.opt = opt
        # build image list
        self.imglist = utils.get_mats_name(opt.baseroot_train)
    
    def __getitem__(self, index):

        # read an image
        namehead = self.imglist[index]
        rgbpath = os.path.join(self.opt.baseroot_train, 'rgb', namehead + '.png')
        matpath = os.path.join(self.opt.baseroot_train, 'mat', namehead + '.mat')
        
        # read a rgb image
        rgb = cv2.imread(rgbpath)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)                  # (H, W, C), uint8
        rgb = rgb / 255.0                                           # (H, W, C), float64

        # read a spectral image
        spectral = h5py.File(matpath)
        spectral = np.transpose(spectral['S'])                      # (H, W, C), float64

        # crop
        if self.opt.crop_size > 0:
            h, w = rgb.shape[:2]
            rand_h = random.randint(0, h - self.opt.crop_size)
            rand_w = random.randint(0, w - self.opt.crop_size)
            rgb = rgb[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]            # (256, 256, 3), in range [0, 1], float64
            spectral = spectral[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]  # (256, 256, 49), in range [0, 1], uint8

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
        self.imglist = utils.get_mats_name(opt.baseroot_val)
    
    def __getitem__(self, index):

        # read an image
        namehead = self.imglist[index]
        rgbpath = os.path.join(self.opt.baseroot_val, 'rgb', namehead + '.png')
        matpath = os.path.join(self.opt.baseroot_val, 'mat', namehead + '.mat')
        
        # read a rgb image
        rgb = cv2.imread(rgbpath)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)                  # (H, W, C), uint8
        rgb = rgb / 255.0                                           # (H, W, C), float64

        # read a spectral image
        spectral = h5py.File(matpath)
        spectral = np.transpose(spectral['S'])                      # (H, W, C), float64

        '''
        # crop
        if self.opt.crop_size > 0:
            h, w = rgb.shape[:2]
            rand_h = random.randint(0, h - self.opt.crop_size)
            rand_w = random.randint(0, w - self.opt.crop_size)
            rgb = rgb[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]            # (256, 256, 3), in range [0, 1], float64
            spectral = spectral[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]  # (256, 256, 49), in range [0, 1], uint8
        '''

        # to tensor
        rgb = torch.from_numpy(rgb.astype(np.float32).transpose(2, 0, 1)).contiguous()
        spectral = torch.from_numpy(spectral.astype(np.float32).transpose(2, 0, 1)).contiguous()

        return rgb, spectral, namehead

    def __len__(self):
        return len(self.imglist)

if __name__ == "__main__":

    def split_images(in_path):
        spectral = h5py.File(in_path)
        print(spectral.keys())
        print(spectral.values())
        print(spectral['S'].shape)
        print(spectral['Wavelengths'].shape)
        print(spectral['Wavelengths'][0])
        data = np.transpose(spectral['S'])
        print(data.shape)
        for i in range(49):
            print(i)
            img = data[:, :, i]
            type_img = img.dtype
            print("type:", type_img)                    # type: float64
            maxnum = np.max(img)
            minnum = np.min(img)
            print("img maxnum", maxnum)
            print("img minnum", minnum)

            a = (img * 255).astype(np.float64)

            maxnum = np.max(a)
            minnum = np.min(a)
            print("a maxnum", maxnum)
            print("a minnum", minnum)

            new_file = './split/' + '%d.jpg'%i
            print(new_file)
            cv2.imwrite(new_file, a)

    in_path = 'F:\\dataset, task related\\Hyperspectral Imaging\\NUS\\train\\rgb\\Scene18.png'
    img = cv2.imread(in_path)
    print(img.shape, img.dtype)

    in_path = 'F:\\dataset, task related\\Hyperspectral Imaging\\NUS\\train\\mat\\Scene18.mat'
    utils.check_path('./split')
    split_images(in_path)
