import argparse
import os
import torch
import numpy as np
import cv2
import hdf5storage as hdf5

import utils
import dataset

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
    parser.add_argument('--data_type', type = str, default = 'arad', help = 'data_type, 5 datasets')
    parser.add_argument('--process_type', type = str, default = 'gen', help = 'process_type, 3 settings')
    parser.add_argument('--network_type', type = str, default = 'awan', help = 'network_type')
    parser.add_argument('--load_name', type = str, default = './models/G_epoch10000_bs4.pth', help = 'load the pre-trained model with certain epoch, None for pre-training')
    parser.add_argument('--test_batch_size', type = int, default = 1, help = 'size of the testing batches for single GPU')
    parser.add_argument('--num_workers', type = int, default = 0, help = 'number of cpu threads to use during batch generation')
    parser.add_argument('--val_path', type = str, default = './validation', help = 'saving path that is a folder')
    parser.add_argument('--enable_patch', type = bool, default = True, help = 'whether using patch at validation')
    parser.add_argument('--patch_size', type = str, default = 256, help = 'patch size at validation')
    # Network initialization parameters
    parser.add_argument('--pad', type = str, default = 'zero', help = 'pad type of networks')
    parser.add_argument('--activ', type = str, default = 'lrelu', help = 'activation type of networks')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type of networks')
    parser.add_argument('--in_channels', type = int, default = 3, help = 'input channels for generator')
    parser.add_argument('--out_channels', type = int, default = 31, help = 'output channels for generator')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'start channels for generator')
    parser.add_argument('--latent_channels', type = int, default = 16, help = 'latent channels for generator')
    # Dataset parameters
    parser.add_argument('--baseroot_train', type = str, default = './dataset/ICVL/train', help = 'baseroot')
    parser.add_argument('--baseroot_val', type = str, default = './dataset/ICVL/val', help = 'baseroot')
    parser.add_argument('--crop_size', type = int, default = 256, help = 'crop size')
    opt = parser.parse_args()
    print(opt)
    
    # ----------------------------------------
    #                   Test
    # ----------------------------------------
    # Initialize
    generator = utils.create_generator(opt).cuda()
    test_dataset = utils.create_dataset_val(opt)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = opt.test_batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True)
    sample_folder = os.path.join(opt.val_path)
    utils.check_path(sample_folder)
    patch_size = opt.patch_size

    # forward
    for i, (img, spectral_img, imgname) in enumerate(test_loader):
        # To device
        img = img.cuda()
        spectral_img = spectral_img.cuda()
        imgname = imgname[0]

        # Forward propagation
        with torch.no_grad():
            if opt.enable_patch:
                _, _, H, W = img.shape
                patchGen = utils.PatchGenerator(H, W, patch_size)
                out = torch.zeros_like(spectral_img)
                for (h, w, top_padding, left_padding, bottom_padding, right_padding) in patchGen.next_patch():
                    img_patch = img[:, :, h:h+patch_size, w:w+patch_size]
                    out_patch = generator(img_patch)
                    out[:, :, h+top_padding:h+patch_size-bottom_padding, w+left_padding:w+patch_size-right_padding] = \
                        out_patch[:, :, top_padding:patch_size-bottom_padding, left_padding:patch_size-right_padding]
            else:
                out = generator(img)

        # Save
        out = out.data.cpu().numpy()[0, :, :, :]
        save_img_name = imgname + '.npy'
        save_img_path = os.path.join(sample_folder, save_img_name)
        np.save(save_img_path, out)
        print('image id:', i, 'image save name:', save_img_name, 'whole numbers of images:', len(test_dataset))
