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
        self.imglist = utils.get_pairs_name(opt.baseroot_train)
    
    def __getitem__(self, index):

        # read an image
        rgbpath = os.path.join(self.opt.baseroot_train, 'generated_rgb', self.imglist[index] + '.png')
        matpath = os.path.join(self.opt.baseroot_train, 'spectral', self.imglist[index] + '.mat')

        # read a rgb image
        rgb = cv2.imread(rgbpath)

        # read a spectral image
        data = h5py.File(matpath)
        data = data['rad']
        data = np.transpose(data, (2, 1, 0))
        spectral = np.zeros((data.shape[1], data.shape[0], data.shape[2]))
        for i in range(31):
            img = data[:, :, i]
            img = np.rot90(img)
            spectral[:, :, i] = img
        
        # crop
        if self.opt.crop_size > 0:
            h, w = spectral.shape[:2]
            rand_h = random.randint(0, h - self.opt.crop_size)
            rand_w = random.randint(0, w - self.opt.crop_size)
            rgb = rgb[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]            # (256, 256, 31), in range [0, 1], float64
            spectral = spectral[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]  # (256, 256, 3), in range [0, 1], float64

        # normalization
        spectral = spectral / 4095.0
        rgb = rgb / 255.0

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
        savename = self.imglist[index]

        # read a rgb image
        rgb = cv2.imread(rgbpath)

        # read a spectral image
        data = h5py.File(matpath)
        data = data['rad']
        data = np.transpose(data, (2, 1, 0))
        spectral = np.zeros((data.shape[1], data.shape[0], data.shape[2]))
        for i in range(31):
            img = data[:, :, i]
            img = np.rot90(img)
            spectral[:, :, i] = img

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
        spectral = spectral / 4095.0
        rgb = rgb / 255.0

        # to tensor
        rgb = torch.from_numpy(rgb.astype(np.float32).transpose(2, 0, 1)).contiguous()
        spectral = torch.from_numpy(spectral.astype(np.float32).transpose(2, 0, 1)).contiguous()

        return rgb, spectral, savename

    def __len__(self):
        return len(self.imglist)

if __name__ == "__main__":

    def split_images(in_path, out_path):
        for fileName in os.listdir(in_path):
            fileName = fileName[0:fileName.find(".")]
            matpath = in_path + fileName +'.mat'
            print(fileName)
            spectral = h5py.File(matpath)
            #print(spectral.keys())
            #print(spectral.values())
            #print(spectral['bands'].shape)
            #print(spectral['rad'].shape)
            #print(spectral['rgb'].shape)

            data = np.transpose(spectral['rad'])
            '''
            for i in range(31):
                print(i)
                img = data[:, :, i]
                type = img.dtype
                #print("type:  ", type)
                a = img.astype(np.float64)
                img90 = np.rot90(a)
                new_file = split_path + fileName + '_%d.jpg' %i
                #print(new_file)
                cv2.imwrite(new_file, img90)
            '''
            rgb = spectral['rgb']
            rgb = np.transpose(rgb)
            #print(rgb)
            #print(rgb.shape)
            rgb = (rgb * 255).astype(np.float64)
            rgb = rgb[:, :, (2, 1, 0)]
            new_file = out_path + fileName + '.png'
            # print(new_file)
            cv2.imwrite(new_file, rgb)

    def read_one_mat(in_path):
        print(in_path)
        spectral = h5py.File(in_path)
        #print(spectral.keys())
        #print(spectral.values())
        #print(spectral['bands'].shape)
        #print(spectral['rad'].shape)
        #print(spectral['rgb'].shape)
        data = spectral['rad']
        print(data.shape)
        print(data.dtype)
        data = np.transpose(data, (2, 1, 0))
        newdata = np.zeros((data.shape[1], data.shape[0], data.shape[2]))
        for i in range(31):
            img = data[:, :, i]
            img = np.rot90(img)
            newdata[:, :, i] = img
        print('newdata.shape', newdata.shape)
        rgb = spectral['rgb']
        print(rgb.shape)
        print(rgb.dtype)
        rgb = np.transpose(rgb, (2, 1, 0))
        '''
        rgb = (rgb * 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        bgr = cv2.resize(bgr, (bgr.shape[1] // 2, bgr.shape[0] // 2))
        print('bgr.shape', bgr.shape)
        cv2.imshow('rgb sample', bgr)
        cv2.waitKey(0)
        '''
        '''
        sample = np.clip(newdata[:, :, 7] / 4095, 0, 1)
        sample = (sample * 255).astype(np.uint8)
        sample = cv2.resize(sample, (sample.shape[1] // 2, sample.shape[0] // 2))
        print('sample.shape', sample.shape)
        cv2.imshow('gray sample', sample)
        cv2.waitKey(0)
        '''
        rgb = (rgb * 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        sample = np.clip(newdata[:, :, 17] / 4095, 0, 1)
        sample = (sample * 255).astype(np.uint8)[:, :, np.newaxis]
        sample = np.concatenate((sample, sample, sample), axis = 2)
        show = np.concatenate((bgr, sample), axis = 1)
        show = cv2.resize(show, (show.shape[1] // 2, show.shape[0] // 2))
        print('show.shape', show.shape)
        cv2.imshow('show sample', show)
        cv2.waitKey(0)

    '''
    in_path = 'C:/Users/a/Desktop/ICVL/train/spectral/'
    out_path = 'C:/Users/a/Desktop/ICVL/train/rgb/'
    split_path = 'C:/Users/a/Desktop/ICVL/train/split/'
    split_images(in_path, out_path)
    '''

    imgpath = 'F:\\dataset, task related\\Hyperspectral Imaging\\ICVL\\train\\spectral\\Lehavim_0910-1627.mat'
    read_one_mat(imgpath)
    