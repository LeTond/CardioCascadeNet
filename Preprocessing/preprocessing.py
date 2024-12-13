# -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.3
Date: 13-12-2024
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""

import sys
import os

import torch
import random 
import time
import cv2
import matplotlib
import pickle
import platform

import nibabel as nib
import pydicom as dicom
import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import statsmodels.api as sm
import matplotlib.pyplot as plt

from torch import nn
from skimage.transform import resize, rescale, downscale_local_mean
from scipy.ndimage import rotate as rotate_image
from torch.utils.data import DataLoader
from sklearn import preprocessing        #pip install scikit-learn
from scipy import ndimage
from configuration import *


class ReadImages():
    def __init__(self, path_to_file):
        self.__path_to_file = path_to_file

    @property
    def path_to_file(self):
        return self.__path_to_file

    @property
    def get_nii(self):
        # matplotlib.use('TkAgg')
        img = nib.load(self.path_to_file)

        return img

    def get_dcm(self):
        origin_dicom = dicom.dcmread(self.path_to_file)
        new_dicom = np.array(origin_dicom.pixel_array)
        
        if len(list(new_dicom.shape)) == 2:
            new_dicom = new_dicom[:, :, np.newaxis]
        else:
            new_dicom = new_dicom.transpose(2, 1, 0)

        return new_dicom

    def get_nii_fov(self):
        img = nib.load(self.path_to_file)
        return img.header.get_zooms()

    @property
    def view_matrix(self):
        # np.set_printoptions(threshold=sys.maxsize)
        return np.array(self.get_nii.dataobj)

    @property
    def get_file_list(self):
        files = os.listdir(self.path_to_file)
        files.sort()
        return files

    def get_file_path_list(self):
        path_list = []

        for root, subfolder, files in os.walk(self.path_to_file):
            for item in files:
                if item.endswith('.nii') or item.endswith('.dcm'):
                    filenamepath = str(os.path.join(root, item))
                    path_list.append(filenamepath)

        return path_list

    def get_dataset_list(self):
        return list(self.get_file_list)


class PreprocessData(MetaParameters):
    def __init__(self, image, mask = None, template = None, names = None, unet_type = None, mask_type = None):
        super().__init__()
        self.__image = image
        self.__mask = mask
        self.__template = template
        self.__names = names
        self.__mask_type = mask_type
        self.__unet_type = unet_type
        self.kernel_size = chklsz.kernel_size(unet_type)

    @property
    def names(self):
        return self.__names

    @property
    def mask_type(self):
        return self.__mask_type

    @property
    def image(self):
        return self.__image

    @property
    def mask(self):
        return self.__mask
   
    @property
    def template(self):
        return self.__template

    @property
    def preprocessing(self):
        image = np.array(self.image, dtype = np.float32)
        image = self.clipping(image)
        image = self.normalization(image)
        image = self.equalization_matrix(matrix = image)
        image = self.rescale_matrix(matrix = image, order = None)
        image = np.array(image.reshape(self.kernel_size, self.kernel_size, 1), dtype = np.float32)

        if self.mask is not None:
            mask = np.array(self.mask, dtype = np.float32)
            mask = self.equalization_matrix(matrix = mask)
            mask = self.rescale_matrix(matrix = mask, order = 0)
            mask = np.array(mask.reshape(self.kernel_size, self.kernel_size, 1), dtype = np.float32)
        else:
            mask = None

        if self.template is not None:
            template = np.array(self.template, dtype = np.float32)

            if self.mask_type != 'infer_bull_level' and self.mask_type != 'train_bull_level':
                template = self.clipping(template)
                template = self.clahe_normalization(template)

            template = self.equalization_matrix(matrix = template)
            template = self.rescale_matrix(matrix = template, order = 0)
            template = np.array(template.reshape(self.kernel_size, self.kernel_size, 1), dtype = np.float32)
        else:
            template = None

        return image, mask, template

    def clipping(self, image):
        image_max = np.max(image)

        if self.CLIP_RATE is not None:
            image = np.clip(image, self.CLIP_RATE[0] * image_max, self.CLIP_RATE[1] * image_max)
        
        return image

    @staticmethod
    def normalization(image):
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        
        return image / np.max(image)

    @staticmethod
    def z_normalization(image):
        mean, std = np.mean(image), np.std(image)
        image = (image - mean) / std
        image += abs(np.min(image))
        
        return image / np.max(image)

    @staticmethod
    def hyst_normalization(image):
        minimum, maximum = 0, 4095
        cur_minimum, cur_maximum = np.min(image), np.max(image)
        
        normalyzed_image = image.copy()
        normalyzed_image = (maximum - minimum) / (cur_maximum - cur_minimum) * (normalyzed_image - cur_minimum) + minimum

        normalyzed_image[normalyzed_image < minimum] = minimum
        normalyzed_image[normalyzed_image > maximum] = maximum

        return normalyzed_image / np.max(normalyzed_image)

    @staticmethod
    def equalize_normalization(image):
        image = image / np.max(image) * 255
        image = image.astype("uint8")
        image = cv2.equalizeHist(image)
        image = image / 255

        return image

    @staticmethod
    def clahe_normalization(image):
        image = image / np.max(image) * 255
        image = image.astype("uint8")
        clahe = cv2.createCLAHE(clipLimit = 4, tileGridSize = (5, 1))
        image = clahe.apply(image)

        return image / np.max(image)

    @staticmethod
    def equalization_matrix(matrix):
        max_kernel = max(matrix.shape[0], matrix.shape[1])
        new_matrix = np.zeros((max_kernel, max_kernel))
        new_matrix[: matrix.shape[0], : matrix.shape[1]] = matrix
        matrix = new_matrix
        
        return matrix

    @staticmethod
    def center_cropping(matrix):
        y, x = matrix.shape
        min_kernel = min(matrix.shape[0], matrix.shape[1])
        startx = (x - min_kernel) // 4 * 3
        starty = (y - min_kernel) // 4 * 3
        
        return matrix[starty:starty + min_kernel, startx:startx + min_kernel]

    def rescale_matrix(self, matrix, order = None):
        shp = matrix.shape
        max_kernel = max(matrix.shape[0], matrix.shape[1])
        scale =  self.kernel_size / max_kernel
        
        return rescale(matrix, (scale, scale), anti_aliasing = False, order = order)

    @property
    def shuff_dataset(self):
        temp = list(zip(self.image, self.mask, self.template, self.names))
        random.shuffle(temp)
        images, masks, templates, names = zip(*temp)
        
        return list(images), list(masks), list(templates), list(names)


class MaskPreprocessing(MetaParameters):
    def __init__(self, image, mask = None, template = None, mask_type = None):    
        super(MetaParameters, self).__init__()
        self.__image = image
        self.__mask = mask
        self.__template = template
        self.__mask_type = mask_type

    @property
    def image(self):
        return self.__image

    @property
    def mask(self):
        return self.__mask
   
    @property
    def template(self):
        return self.__template
    
    @property
    def mask_type(self):
        return self.__mask_type

    @property
    def lv_preprocessing(self):
        if self.mask_type == 'bgcropp':
            template = self.mask.copy()
            template[template != 0] = 1
            image = self.image.copy() * template

        elif self.mask_type == 'lvcropp':
            template = self.mask.copy()
            template[template != 1] = 2
            template[template == 1] = 0
            template[template == 2] = 1
            image = self.image.copy() * template

        elif self.mask_type == 'bglvcropp':
            template = self.mask.copy()
            template[template == 1] = 0
            template[template != 0] = 1
            image = self.image.copy() * template

        else:
            image = self.image.copy()
            image = np.array(image.reshape(image.shape[0], image.shape[1], 1), dtype = np.float32)
        
        return image, self.mask, self.template
    
    @property
    def choose_mask_preprocessing(self):
        if self.mask_type == 'train_bull_level':
            return self.train_bull_level_preprocessing

        elif self.mask_type == 'infer_bull_level':
            return self.infer_bull_level_preprocessing

        elif self.mask_type == 'myo_level':
            return self.myo_level_preprocessing
        
        elif self.mask_type is None:
            return self.image, self.mask, self.template
        
        else:
            return self.lv_preprocessing

    @property
    def mask_preprocessing(self):
        return self.choose_mask_preprocessing

    @property
    def myo_level_preprocessing(self):
        mask = self.mask.copy()

        for basal in range(1, 7):
            if (mask == basal).any():
                mask[mask == basal] = 1
                mask[mask != basal] = 1
        for medial in range(7, 13):
            if (mask == medial).any():
                mask[mask == medial] = 2
                mask[mask != medial] = 2
        for apical in range(13, 17):
            if (mask == apical).any():
                mask[mask == apical] = 3
                mask[mask != apical] = 3
        for apex in range(17, 18):
            if (mask == apex).any():
                mask[mask == apex] = 4
                mask[mask != apex] = 4

        return self.image, mask, self.template

    @property
    def train_bull_level_preprocessing(self):
        template = self.template.copy()
        mask = self.mask.copy()

        template[template == 0] = 40
        template[template != 40] = 60

        for basal in range(1, 7):
            if (mask == basal).any():
                template[template == 40] = 1
        for medial in range(7, 13):
            if (mask == medial).any():
                template[template == 40] = 2
        for apical in range(13, 17): 
            if (mask == apical).any():
                template[template == 40] = 3
        for apex in range(17, 18):
            if (mask == apex).any():
                template[template == 40] = 4

        template[template == 60] = 0
        template = template / len(self.MYOLEVEL_DICT_CLASS)

        return self.image, self.mask, template

    @property
    def infer_bull_level_preprocessing(self):
        mask = self.mask.copy()                 #Unet4_mask MYO_LEVEL
        template = self.template.copy()         #Unet3_mask SCAR

        template[template == 1] = 0
        template[template > 0] = 60

        myo_lvl = round(np.max(mask))        
        if self.template[self.template == 1].sum().item() == 0:
            myo_lvl = 4

        template[template == 0] = myo_lvl
        template[template == 60] = 0

        template = template / len(self.MYOLEVEL_DICT_CLASS)

        return self.image, self.mask, template


class Augmentation(MetaParameters):
    def __init__(self, image, mask = None, template = None, unet_type = None):
        super().__init__()
        self.__image = image
        self.__mask = mask
        self.__template = template
        self.__unet_type = unet_type
        self.kernel_size = chklsz.kernel_size(unet_type)

    @property
    def unet_type(self):
        return self.__unet_type

    @property
    def image(self):
        return self.__image

    @property
    def mask(self):
        return self.__mask
   
    @property
    def template(self):
        return self.__template

    @property
    def define_kernel_size(self):
        return self.image.shape[0]

    @property
    def angle_list(self):
        if self.AUGMENTATION:
            angle_list = list(set([random.choice([0, 90, 180, 270]) for i in range(3)]))
        else: 
            angle_list = [0]

        return angle_list

    @property
    def angle(self):
        return random.choice(self.angle_list)

    @property
    def rotate_2d(self):
        angle = self.angle
        
        image = rotate_image(self.image, angle)
        mask = rotate_image(self.mask, angle)

        if self.template is not None:
            template = rotate_image(self.template, angle)
        else:
            template = None

        return image, mask, template

    @property
    def choose_noise(self):
        return random.choice(['with', 'without'])

    @property
    def gauss_noise(self):
        choose_noise = self.choose_noise
        sigma, mean = 2, 0.5
        
        if choose_noise == 'with':
            try:
                noise = np.random.normal(mean, sigma ** 0.5, self.image.shape)
                noisy_image = self.image + noise
                noisy_image2 = self.template + noise

                noisy_image = np.array(noisy_image, dtype = np.float32)
                noisy_image2 = np.array(noisy_image2, dtype = np.float32)
                
            except:
                print('Gauss Noise Application Error')

            return noisy_image, self.mask, noisy_image2
        
        else:
            return self.image, self.mask, self.template

    @property
    def rician_noise(self):
        choose_noise = self.choose_noise

        if choose_noise == 'with':
            random_v = random.choice([10, 20, 5, 25])
            random_s = random.choice([50, 25, 30, 70])
            num_samples = self.kernel_size * self.kernel_size
        
            try:
                noise = np.random.normal(scale = random_s, size = (num_samples, 2)) + [[random_v, 0]]
                noise = np.linalg.norm(noise, axis = 1)
                noise = np.array(noise.reshape(self.kernel_size, self.kernel_size, 1), dtype = np.float32)
                
                noisy_image = self.image + noise
                noisy_image2 = self.template + noise

                noisy_image = np.array(noisy_image, dtype = np.float32)
                noisy_image2 = np.array(noisy_image2, dtype = np.float32)

            except:
                print('Rician Noise Application Error')
            
            return noisy_image, self.mask, noisy_image2
        
        else:
            return self.image, self.mask, self.template


class CroppPreprocessData(MetaParameters):
    def __init__(self, images = None, masks = None, templates = None, unet_type = None):
        super(MetaParameters, self).__init__()
        self.__images = images
        self.__masks = masks
        self.__templates = templates
        self.__unet_type = unet_type

    @property
    def images(self):
        return self.__images

    @property
    def masks(self):
        return self.__masks
   
    @property
    def templates(self):
        return self.__templates

    @property
    def unet_type(self):
        return self.__unet_type

    def presegmentation_tissues(self, def_coord, gap_1 = None):
        list_top, list_bot, list_left, list_right = [], [], [], []
        list_weight_mass_x, list_weight_mass_y = [], []

        shp = self.images.shape
        count = 0
        gap = self.CROPP_KERNEL // 2

        last_top, last_bot, last_left, last_right = \
        (shp[0] // 2 - gap), (shp[1] // 2 - gap), (shp[0] // 2 + gap), (shp[1] // 2 + gap)

        for slc in range(shp[2]):
            image = self.images[:, :, slc]
            mask = self.masks[:, :, slc]

            if (mask != 0).any():
                count += 1
                # weight_mass_y, weight_mass_x = ndimage.measurements.center_of_mass(mask)
                # list_weight_mass_y.append(round(weight_mass_y))
                # list_weight_mass_x.append(round(weight_mass_x))  

                predict_mask = np.where(mask != 0)
                last_top = np.min(predict_mask[0])
                last_bot = np.max(predict_mask[0])
                last_left = np.min(predict_mask[1])
                last_right = np.max(predict_mask[1])
            else:
                # list_weight_mass_y.append((last_top+last_bot)//2)
                # list_weight_mass_x.append((last_left+last_right)//2)  
                count += 1

            list_top.append(last_top)
            list_bot.append(last_bot)
            list_left.append(last_left)
            list_right.append(last_right)

        mean_top = np.array(list_top).sum() // count
        mean_left = np.array(list_left).sum() // count
        mean_bot = np.array(list_bot).sum() // count 
        mean_right = np.array(list_right).sum() // count

        if def_coord is None:
            # center_row = np.array(list_weight_mass_y).sum() // count
            # center_column = np.array(list_weight_mass_x).sum() // count
            center_row = (mean_bot + mean_top) // 2
            center_column = (mean_left + mean_right) // 2
        
        else:
            center_row, center_column = def_coord

        if type(gap_1) == list:
            center_row = center_row + gap_1[0]
            center_column = center_column + gap_1[1]

        if self.unet_type == 'close_cropp':
            for slc in range(shp[2]):
                image_template = np.zeros((shp[0], shp[1])).copy()

                if list_top[slc] == (shp[0] // 2 - gap) and list_bot[slc] == (shp[1] // 2 - gap) \
                and list_left[slc] == (shp[0] // 2 + gap) and list_right[slc] == (shp[1] // 2 + gap):
                    image_template[center_row - 2 * gap_1 : center_row + 2 * gap_1, \
                    center_column - 2 * gap_1 : center_column + 2 * gap_1] = 1
                
                elif list_top[slc] > mean_bot and list_bot[slc] > mean_bot \
                and list_left[slc] > mean_bot and list_right[slc] > mean_bot:
                    image_template[center_row - 2 * gap_1 : center_row + 2 * gap_1, \
                    center_column - 2 * gap_1 : center_column + 2 * gap_1] = 1
                
                elif list_top[slc] > mean_right and list_bot[slc] > mean_right \
                and list_left[slc] > mean_right and list_right[slc] > mean_right:
                    image_template[center_row - 2 * gap_1 : center_row + 2 * gap_1, \
                    center_column - 2 * gap_1 : center_column + 2 * gap_1] = 1
                
                elif list_top[slc] < mean_top and list_bot[slc] < mean_top \
                and list_left[slc] < mean_top and list_right[slc] < mean_top:
                    image_template[center_row - 2 * gap_1 : center_row + 2 * gap_1, \
                    center_column - 2 * gap_1 : center_column + 2 * gap_1] = 1
                
                elif list_top[slc] < mean_left and list_bot[slc] < mean_left \
                and list_left[slc] < mean_left and list_right[slc] < mean_left:
                    image_template[center_row - 2 * gap_1 : center_row + 2 * gap_1, \
                    center_column - 2 * gap_1 : center_column + 2 * gap_1] = 1
                
                else:
                    image_template[list_top[slc] - gap_1 : list_bot[slc] + gap_1, \
                    list_left[slc] - gap_1 : list_right[slc] + gap_1] = 1

                self.images[:, :, slc] = self.images[:, :, slc] * image_template

        images = self.images[center_row - gap: center_row + gap, center_column - gap: center_column + gap, :]
        masks = self.masks[center_row - gap: center_row + gap, center_column - gap: center_column + gap, :]
        
        if self.templates is not None: 
            templates = self.templates[center_row - gap: center_row + gap, center_column - gap: center_column + gap, :]
        
        else:
            templates = None

        return images, masks, templates, [center_row, center_column]


class ViewData():
    def view_img(self, img):
        width, height, queue = img.shape
        array_data = np.arange(24, dtype = np.int16).reshape((2, 3, 4))
        print(width, height, queue)
        num = 1
        for i in range(0, queue, 1):
            img_arr = img.dataobj[:, :, i]
            plt.subplot(4, 5, num)
            plt.imshow(img_arr, cmap = 'gray')
            num += 1
        plt.show()



