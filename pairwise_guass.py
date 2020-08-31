from __future__ import absolute_import, division

import numpy as np
from collections import namedtuple
from torch.utils.data import Dataset
from torchvision.transforms import Compose, CenterCrop, RandomCrop, ToTensor
from PIL import Image, ImageStat, ImageOps
import cv2
import random
from scipy import signal
import matplotlib.pyplot as plt

import skimage


class RandomStretch(object):

    def __init__(self, max_stretch=0.05, interpolation='bilinear'):
        assert interpolation in ['bilinear', 'bicubic']
        self.max_stretch = max_stretch
        self.interpolation = interpolation

    def __call__(self, img):
        scale = 1.0 + np.random.uniform(
            -self.max_stretch, self.max_stretch)
        size = np.round(np.array(img.size, float) * scale).astype(int)
        if self.interpolation == 'bilinear':
            method = Image.BILINEAR
        elif self.interpolation == 'bicubic':
            method = Image.BICUBIC
        return img.resize(tuple(size), method)


class Pairwise(Dataset):

    def __init__(self, seq_dataset, **kargs):
        super(Pairwise, self).__init__()
        self.cfg = self.parse_args(**kargs)

        self.seq_dataset = seq_dataset
        self.indices = np.random.permutation(len(seq_dataset))
        # augmentation for exemplar and instance images
        self.transform_z = Compose([
            RandomStretch(max_stretch=0.05),
            CenterCrop(self.cfg.instance_sz - 8),
            RandomCrop(self.cfg.instance_sz - 2 * 8),
            CenterCrop(self.cfg.exemplar_sz),
            ToTensor()])
        self.transform_x = Compose([
            RandomStretch(max_stretch=0.05),
            CenterCrop(self.cfg.instance_sz - 8),
            RandomCrop(self.cfg.instance_sz - 2 * 8),
            ToTensor()])

    def parse_args(self, **kargs):
        # default parameters
        cfg = {
            'pairs_per_seq': 15,
            'max_dist': 100,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            'guass_std': 25}

        for key, val in kargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('GenericDict', cfg.keys())(**cfg)

    def __getitem__(self, index):
        index = self.indices[index % len(self.seq_dataset)]
        img_files, anno = self.seq_dataset[index]

        # remove too small objects
        valid = anno[:, 2:].prod(axis=1) >= 10
        img_files = np.array(img_files)[valid]
        anno = anno[valid, :]

        rand_z, rand_x = self._sample_pair(len(img_files))

        exemplar_image = Image.open(img_files[rand_z])
        exemplar_img = self._crop_and_resize(exemplar_image, anno[rand_z])
        exemplar_image = 255.0 * self.transform_z(exemplar_img)
        width, height = exemplar_img.size

        exemplar_noise = self.guass_noise(exemplar_img, 0.09)
        exemplar_noise = 255.0 * self.transform_z(exemplar_noise)

        instance_image = Image.open(img_files[rand_x])
        instance_img = self._crop_and_resize(instance_image, anno[rand_x])
        instance_image = 255.0 * self.transform_x(instance_img)
        
        width, height = instance_img.size

        instance_noise = self.guass_noise(instance_img, 0.09)#self.gaussian_map(instance_img, width, self.cfg.guass_std)
        
        instance_noise = 255.0 * self.transform_x(instance_noise)

        return exemplar_image, exemplar_noise, instance_image, instance_noise

    def __len__(self):
        return self.cfg.pairs_per_seq * len(self.seq_dataset)
    
    
    def guass_noise(self, image, var):
        '''gaussian noise'''
        image = np.array(image)
        row,col,ch= image.shape
        #plt.imshow(image)
        #plt.show()
        mean = 0
        #var = 0.1
        #================Guassian 0.09######################
        sigma = 0.09
        #noise = np.random.normal(0.0, 0.9, image.shape)
        #gauss = np.random.normal(mean,sigma**0.5,(row,col,ch))
        #gauss = gauss.reshape(row,col,ch)
        #plt.imshow(gauss)
        #plt.show()
        # noisy = image + noise
        # plt.imshow(noisy)
        # plt.show()
        # noisy = Image.fromarray(noisy.astype('uint8'))
        # plt.imshow(noisy)
        # plt.show()
        gimg = skimage.util.random_noise(image, mode="gaussian", var = sigma )
#        plt.imshow(gimg)
#        plt.show()

        output = Image.fromarray(gimg)
        plt.imshow(np.array(output).astype('uint8'))
        plt.show()

        return noisy 
    def gaussian_map(self,image, kernlen, std):
        """Returns a 3D Gaussian map."""
        
        gkernel_1d = signal.gaussian(kernlen, std)
        
       
        gkernel_2d = np.outer(gkernel_1d, gkernel_1d)
    
        #add in the depth for channel
        gkernel_2d_chan1 = np.dstack([gkernel_2d, gkernel_2d])
        gkernel_2d_chan3 = np.dstack([gkernel_2d_chan1, gkernel_2d])
        w, h, c = gkernel_2d_chan3.shape
        print(w, h, c)
        plt.imshow(gkernel_2d_chan3)
        plt.show()
        
        
        
        #generating weighted g_Map
       
        gkernel_2d_chan3 = np.multiply(gkernel_2d_chan3, image).astype('uint8')
        plt.imshow(image)
        plt.show()
        plt.imshow(gkernel_2d_chan3)
        plt.show()
#        
        image_final= np.array(gkernel_2d_chan3).astype('uint8') + np.array(image).astype('uint8')
        plt.imshow(image)
        plt.show()
        plt.imshow(image_final.astype('uint8'))
        plt.show()
#        image_test= np.array(image).astype('uint8')
#        image_final_test = image_final.astype('uint8')
        image_final = gkernel_2d_chan3
        output = Image.fromarray(image_final)
        
        return output


    def _sample_pair(self, n):
        assert n > 0
        if n == 1:
            return 0, 0
        elif n == 2:
            return 0, 1
        else:
            max_dist = min(n - 1, self.cfg.max_dist)
            rand_dist = np.random.choice(max_dist) + 1
            rand_z = np.random.choice(n - rand_dist)
            rand_x = rand_z + rand_dist

        return rand_z, rand_x

    def _crop_and_resize(self, image, box):
        # convert box to 0-indexed and center based
        box = np.array([
            box[0] - 1 + (box[2] - 1) / 2,
            box[1] - 1 + (box[3] - 1) / 2,
            box[2], box[3]], dtype=np.float32)
        center, target_sz = box[:2], box[2:]

        # exemplar and search sizes
        context = self.cfg.context * np.sum(target_sz)
        z_sz = np.sqrt(np.prod(target_sz + context))
        x_sz = z_sz * self.cfg.instance_sz / self.cfg.exemplar_sz

        # convert box to corners (0-indexed)
        size = round(x_sz)
        corners = np.concatenate((
            np.round(center - (size - 1) / 2),
            np.round(center - (size - 1) / 2) + size))
        corners = np.round(corners).astype(int)

        # pad image if necessary
        pads = np.concatenate((
            -corners[:2], corners[2:] - image.size))
        npad = max(0, int(pads.max()))
        if npad > 0:
            avg_color = ImageStat.Stat(image).mean
            # PIL doesn't support float RGB image
            avg_color = tuple(int(round(c)) for c in avg_color)
            image = ImageOps.expand(image, border=npad, fill=avg_color)

        # crop image patch
        corners = tuple((corners + npad).astype(int))
        patch = image.crop(corners)

        # resize to instance_sz
        out_size = (self.cfg.instance_sz, self.cfg.instance_sz)
        patch = patch.resize(out_size, Image.BILINEAR)
        #print("patch",patch)

        return patch
