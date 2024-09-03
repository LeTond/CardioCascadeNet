 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1
Date: 21-07-2024
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""


import torch
import random
import sys
import time
import cv2
import matplotlib
import os
import pickle
import platform

import nibabel as nib
import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchvision.transforms as transforms
import statsmodels.api as sm
import matplotlib.pyplot as plt
import albumentations as A

from torch import nn
from torch.utils.data import DataLoader
from sklearn import preprocessing  # pip install scikit-learn

from Training.dataset import MyDataset
from Training.ranger import Ranger
from Training.optimizer import Lion

from parameters import MetaParameters
from Preprocessing.dirs_logs import create_dir, create_dir_log, log_stats
from Model.unet2D import UNet_2D, UNet_2D_AttantionLayer, UNetResnet, SegNet
from Model.unet3D import UNet_3D, UNet_3D_AttantionLayer
# from Model.FCT.utils.model import FCT
# from Model.resnet import ResNet, BasicBlock
# from Model.models import bounding_box_CNN


########################################################################################################################
# Show software and harware
########################################################################################################################
print(f"Python Platform: {platform.platform()}")
print(f'python version: {sys.version}')
print(f'torch version: {torch.__version__}')
print(f'numpy version: {np.__version__}')
print(f'pandas version: {pd.__version__}')



########################################################################################################################
# Choose device
########################################################################################################################
global device


class ChooseDevice:
    @staticmethod
    def _device():
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        return device

    @property
    def device(self):
        return self._device()


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight = None, gamma = 2,reduction = 'mean'):    #reduction='sum'
        super(FocalLoss, self).__init__(weight,reduction = reduction)
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target,reduction = self.reduction,weight = self.weight)
        pt = torch.exp( - ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss
        

device = ChooseDevice().device
# device = torch.device('cpu')
print(device)


########################################################################################################################
# COMMENTS
########################################################################################################################
meta = MetaParameters()

create_dir_log(meta.PROJ_NAME)

if meta.UNET3 is True:
    model_key = meta.UNET3_FOLD

elif meta.UNET2 is True and meta.UNET3 is False:
    model_key = meta.UNET2_FOLD

elif meta.UNET2 is False and meta.UNET3 is False:
    model_key = meta.UNET1_FOLD

if meta.PRETRAIN:    
    try:
        # model = torch.load(f'{projec_name}/{meta.MODEL_NAME}.pth').to(device=device)
        # checkpoint = torch.load(f'{meta.PROJ_NAME}/{meta.DATASET_NAME}_model.pth', map_location=torch.device('cpu')).to(device=device)
        checkpoint = torch.load(f'{meta.PROJ_NAME}/{meta.DATASET_NAME}_model.pth').to(device = device)
        checkpoint = checkpoint[f'Net_{meta.DATASET_NAME}_{model_key}']
        model = checkpoint['Model']
        model.load_state_dict(checkpoint['weights'])        
        # model.eval()
        print(f'model loaded: {meta.DATASET_NAME}/{meta.MODEL_NAME}.pth')
        # print(f'model loaded: {projec_name}/{meta.MODEL_NAME}.pth')
    except:
        print('no trained models')
        model = UNet_2D_AttantionLayer().to(device = device)
else:
    model = UNet_2D_AttantionLayer().to(device = device)
    # model = UNet_2D().to(device=device)
    # model = UNetResnet().to(device=device)
    # model = SegNet().to(device=device)


# from torchsummary import summary

# summary(model,input_size=(1,512, 512))


if meta.FREEZE_BN is True:
    for name, child in model.named_children(): 
        if name in ['decoder2', 'decoder1', 'upconv1', 'upconv2', 'conv', 'Att2', 'Att1']: 
        # if name in ['decoder1', 'upconv1', 'conv', 'Att1']: 
        # if name in ['decoder1']: 
            print(name + ' has been unfrozen.') 
            for param in child.parameters(): 
                param.requires_grad = True 
        else: 
            for param in child.parameters(): 
                param.requires_grad = False

loss_function = nn.CrossEntropyLoss(weight = meta.CE_WEIGHTS).to(device)
# loss_function = nn.CrossEntropyLoss().to(device)
# loss_function = FocalLoss(weight = meta.CE_WEIGHTS).to(device)
# loss_function = FocalLoss().to(device)

# optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr = meta.LR, weight_decay = meta.WDC)
# optimizer = torch.optim.Adam(model.parameters(), lr = meta.LR, weight_decay = meta.WDC)
# optimizer = torch.optim.AdamW(model.parameters(), lr = meta.LR, weight_decay = meta.WDC)
# optimizer = Lion(model.parameters(), lr = meta.LR, betas = (0.9, 0.99), weight_decay = meta.WDC)
# optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay = wdc, amsgrad = False)
# optimizer = torch.optim.SGD(model.parameters(), lr = meta.LR, weight_decay = meta.WDC, momentum = 0.9, nesterov = True)
optimizer = Ranger(model.parameters(), lr = meta.LR, k = 6, N_sma_threshhold = 5, weight_decay = meta.WDC)

scheduler_gen = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max = meta.TMAX, eta_min = 0, last_epoch = -1, verbose = True)

# scheduler_gen = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode = 'min', factor = 0.8, patience = 5, threshold = 0.0001, threshold_mode = 'rel', 
#     cooldown = 0, min_lr = 0, eps = 1e-08, verbose = 'deprecated')

########################################################################################################################
## Main image transforms in Dataloder
########################################################################################################################
default_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

transform_01 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation((-10, 10), expand = False),
    transforms.RandomHorizontalFlip(0.7),
    transforms.RandomVerticalFlip(0.7),
    transforms.ToTensor(),
])

transform_02 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomVerticalFlip(1.0),
    transforms.ToTensor(),
])

transform_03 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(1.0),
    transforms.ToTensor(),
])

transform_04 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomAffine(degrees = (-2, 2), translate = (0.05, 0.25), scale = (0.75, 1.25)),
    transforms.ToTensor(),
])

def aug_transforms():
    return [
        # A.VerticalFlip(p=1),
        # A.HorizontalFlip(p=1),
        # A.Rotate(limit=(-90, 90), interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, value=None, mask_value=None,
                 # always_apply=False, p=1),
        A.ElasticTransform(alpha = 20, sigma = 50, alpha_affine = 8,
                           interpolation = cv2.INTER_NEAREST, border_mode = cv2.BORDER_CONSTANT, value = None,
                           mask_value = None, always_apply = False, approximate = False, p = 1),

        # A.Crop(0, 40, 100, 144),
        # A.Crop(40, 40, 140, 144),
        # A.Crop(0, 0, 144, 144),
        # A.Crop(10, 20, 144, 144)
        
        # A.RandomBrightnessContrast()
        # A.GridDistortion(num_steps = 20, distort_limit = 0.2, interpolation = cv2.INTER_NEAREST,
        #                     border_mode = cv2.BORDER_CONSTANT, value = None, mask_value = None,
        #                     always_apply = False, p = 1)

    ]

transform_05 = A.Compose(A.ElasticTransform(alpha = 20, sigma = 50, alpha_affine = 8,
                           interpolation = cv2.INTER_NEAREST, border_mode = cv2.BORDER_CONSTANT, value = None,
                           mask_value = None, always_apply = False, approximate = False, p = 1))

transform_06 = A.Compose(A.GridDistortion(num_steps = 10, distort_limit = 0.05, interpolation = cv2.INTER_NEAREST,
                            border_mode = cv2.BORDER_CONSTANT, value = None, mask_value = None,
                            always_apply = False, p = 1))



# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.RandomRotation((-15, 15), expand=False),
#     transforms.RandomHorizontalFlip(0.5),
#     transforms.RandomVerticalFlip(0.5),
#     # transforms.GaussianBlur(19), 
#     # transforms.RandomResizedCrop(meta.KERNEL),
#     transforms.RandomAffine(degrees=(-45, 45), translate=(0.05, 0.15), scale=(0.75, 1.5)),
#     # transforms.RandomPerspective(distortion_scale=0.7, p=1, interpolation=2, fill=0),
#     # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
#     # transforms.RandomCrop(meta.KERNEL//2, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'),
#     # transforms.Resize((meta.KERNEL, meta.KERNEL)),
#     transforms.ToTensor(),
# ])


# transforms_list = [transform_01, transform_02, transform_03, transform_04, transform_05]

# from torchsummary import summary
# device = 'cpu'
# model = UNet_2D_AttantionLayer().to(device)
# summary(model,input_size=(1,256,256))

