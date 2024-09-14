 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.1
Date: 03-09-2024
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

from Preprocessing.dirs_logs import FileDirectoryWorker


########################################################################################################################
# Show software and harware
########################################################################################################################
print(f"Python Platform: {platform.platform()}")
print(f'python version: {sys.version}')
print(f'torch version: {torch.__version__}')
print(f'numpy version: {np.__version__}')
print(f'pandas version: {pd.__version__}')


class MetaParameters:
    FREEZE_BN = True
    PRETRAIN = True
    AUGMENTATION = True
    NOISE = True
    EMPTY = True
    MULTYGAP = True
    UNET1 = True
    UNET2 = True
    UNET3 = True
    UNET4 = True
    UNET5 = True
    BGCROPP = True
    LVCROPP = True
    BGLVCROPP = True

    # AUGMENTATION = False
    FREEZE_BN = False
    PRETRAIN = False
    NOISE = False
    EMPTY = False
    MULTYGAP = False
    # UNET1 = False
    # UNET2 = False
    # UNET3 = False
    UNET4 = False
    UNET5 = False
    BGCROPP = False
    LVCROPP = False
    BGLVCROPP = False

    """ Network configuration """
    KERNEL = 192
    CROPP_KERNEL = 64
    CHANNELS = 2
    LR = 1e-3
    BT_SZ = 32      # 512 x 512 - max batch_size == 8 if features == 32
    EPOCHS = 1000
    DROPOUT = 0.2
    FEATURES = 16   # 32 - 
    WDC = 1e-4
    EARLY_STOPPING = 50
    TMAX = 50
    SHUFFLE = True
    # CLIP_RATE = [0.1, 0.9]
    CLIP_RATE = None
    
    SCAR_DICT_CLASS = {
                0: "Background", 1: "LV", 2: "MYO", 3: "FIB", # 4: "No_reflow", # 5: "Tromb",
                }

    MYOLEVEL_DICT_CLASS = {
              0: "Background", 1: "Basal", 2: "Medial", 3: "Apical", 4: "Apex",}

    BULLEYE_DICT_CLASS = {
                    0: "Background", 
                    1: "Seg 01", 2: "Seg 02", 3: "Seg 03", 4: "Seg 04", 5: "Seg 05", 6: "Seg 06",
                    7: "Seg 07", 8: "Seg 08", 9: "Seg 09", 10: "Seg 10", 11: "Seg 11", 12: "Seg 12",
                    13: "Seg 13", 14: "Seg 14", 15: "Seg 15", 16: "Seg 16",
                    17: "Seg 17",
                    }

    TARGET_CE_WEIGHTS = torch.FloatTensor([0.1, 0.5, 0.7, 0.9])
    SCAR_CE_WEIGHTS = torch.FloatTensor([0.4, 0.7, 0.5, 0.9])
    MYOLEVEL_CE_WEIGHTS = torch.FloatTensor([0.1, 0.8, 0.7, 0.9, 0.99])
    BULLEYE_CE_WEIGHTS = torch.FloatTensor(
        [0.1, 
        0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 
        0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 
        0.8, 0.8, 0.8, 0.8, 
        1.2])

    DICT_CLASS = SCAR_DICT_CLASS
    # DICT_CLASS = MYOLEVEL_DICT_CLASS
    # DICT_CLASS = BULLEYE_DICT_CLASS
    NUM_CLASS = len(DICT_CLASS)

    CE_WEIGHTS = TARGET_CE_WEIGHTS
    # CE_WEIGHTS = SCAR_CE_WEIGHTS
    # CE_WEIGHTS = MYOLEVEL_CE_WEIGHTS
    # CE_WEIGHTS = BULLEYE_CE_WEIGHTS

    """ Project configuration """
    FOLD_NAME = 'full'  # [01:05] xor full
    DATASET_NAME = 'ALMAZ'
    # DATASET_NAME = 'BULLEYE'
    # DATASET_NAME = 'HCM_adult'
    
    MODEL_NAME = 'model_best'
    DATASET_DIR = './Dataset/'

    ## NEW DIRECTNAMES
    UNET1_FOLD = f'Unet1_Fold_{FOLD_NAME}/'
    UNET2_FOLD = f'Unet2_Fold_{FOLD_NAME}/'
    UNET3_FOLD = f'Unet3_Fold_{FOLD_NAME}/'
    UNET4_FOLD = f'Unet4_Fold_{FOLD_NAME}/'
    UNET5_FOLD = f'Unet5_Fold_{FOLD_NAME}/'

    PROJ_NAME = f'./Results/{DATASET_NAME}'

    ORIGS_DIR = f'{DATASET_DIR}{DATASET_NAME}_origin'
    MASKS_DIR = f'{DATASET_DIR}{DATASET_NAME}_mask'

    NEW_DATA_PATH = f'./Dataset/{DATASET_NAME}_origin_new/'
    NEW_UNET1_MASK_PATH = f'./Dataset/{DATASET_NAME}_Unet1_mask_new/'
    NEW_UNET2_MASK_PATH = f'./Dataset/{DATASET_NAME}_Unet2_mask_new/'
    NEW_UNET3_MASK_PATH = f'./Dataset/{DATASET_NAME}_Unet3_mask_new/'
    NEW_UNET4_MASK_PATH = f'./Dataset/{DATASET_NAME}_Unet4_mask_new/'
    NEW_UNET5_MASK_PATH = f'./Dataset/{DATASET_NAME}_Unet5_mask_new/'


class ChooseDevice:
    def __init__(self):    
        self.print_device

    @staticmethod
    def __device():
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        return device

    @property
    def device(self):
        return self.__device()

    @property
    def print_device(self):
        print(self.device)


global device


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
        

class ChooseKernelSize(MetaParameters):
    def __init__(self):    
        super().__init__()

    def matrix_size(self, unet_type = None):
        if unet_type is None:
            if self.UNET5 is True:
                return 'cropp'
            elif self.UNET4 is True and self.UNET5 is False:
                return 'cropp'
            elif self.UNET3 is True and self.UNET4 is False:
                return 'close_cropp'
            elif self.UNET2 is True and self.UNET3 is False:
                return 'cropp'
            elif self.UNET1 is True and self.UNET2 is False:
                return 'default'
            else: 
                raise ValueError("Check UNET configuration. Make shure that all UNET MetaParameters before target UNET(N) is True")
        else:
            return unet_type

    def kernel_size(self, unet_type = None):
        matrix_size = self.matrix_size(unet_type)

        if matrix_size == 'default':
            return self.KERNEL
        elif matrix_size == 'cropp':
            return self.CROPP_KERNEL
        elif matrix_size == 'close_cropp':
            return self.CROPP_KERNEL
        else:
            raise ValueError("Check matrix size value. Make shure that input value is correct for chosen matrix preprocessing")


class ChooseModelConfig(MetaParameters):
    def __init__(self):    
        super(MetaParameters, self).__init__()
        self.__model = self.choose_train_model
  
    @property
    def model(self):
        return self.__model

    @property  
    def choose_model_key(self):
        if self.UNET5 is True:
            return self.UNET5_FOLD
        elif self.UNET4 is True and self.UNET5 is False:
            return self.UNET4_FOLD
        elif self.UNET3 is True and self.UNET4 is False:
            return self.UNET3_FOLD
        elif self.UNET2 is True and self.UNET3 is False:
            return self.UNET2_FOLD
        elif self.UNET1 is True and self.UNET2 is False:
            return self.UNET1_FOLD
        else:
            raise ValueError("Check UNET configuration. Make shure that all UNET MetaParameters before target UNET(N) is True")

    @property
    def model_key(self):
        return self.choose_model_key

    @property
    def choose_train_model(self):
        if self.PRETRAIN:    
            try:
                checkpoint = torch.load(f'{self.PROJ_NAME}/{self.DATASET_NAME}_model.pth')
                checkpoint = checkpoint[f'Net_{self.DATASET_NAME}_{self.model_key}']

                # checkpoint = torch.load(f'./Results/ALMAZ/ALMAZ_model.pth')
                # checkpoint = checkpoint[f'Net_ALMAZ_{self.model_key}']
                
                model = checkpoint['Model']
                model.load_state_dict(checkpoint['weights'])  
                model.eval()      
                print(f'model loaded: {self.DATASET_NAME}/{self.MODEL_NAME}.pth')

            except:
                print('no trained models')
                model = UNet_2D_AttantionLayer().to(device = device)
        else:
            model = UNet_2D_AttantionLayer().to(device = device)
            # model = UNet_2D().to(device=device)
            # model = UNetResnet().to(device=device)
            # model = SegNet().to(device=device)

        return model

    @property
    def freeze_model_bn(self):
        if self.FREEZE_BN is True:
            for name, child in self.model.named_children(): 
                if name in ['encoder4', 'bottleneck', 'decoder4', 'decoder3','decoder2', 'decoder1', 'upconv4', 'upconv3', 'upconv2', 'upconv1', 'conv', 'Att4', 'Att3', 'Att2', 'Att1']:
                    print(name + ' has been unfrozen.') 
                    for param in child.parameters(): 
                        param.requires_grad = True 
                else: 
                    for param in child.parameters(): 
                        param.requires_grad = False

        # optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr = meta.LR, weight_decay = meta.WDC)
        # optimizer = torch.optim.Adam(model.parameters(), lr = meta.LR, weight_decay = meta.WDC)
        # optimizer = torch.optim.AdamW(model.parameters(), lr = meta.LR, weight_decay = meta.WDC)
        # optimizer = Lion(model.parameters(), lr = meta.LR, betas = (0.9, 0.99), weight_decay = meta.WDC)
        # optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay = wdc, amsgrad = False)
        # optimizer = torch.optim.SGD(model.parameters(), lr = meta.LR, weight_decay = meta.WDC, momentum = 0.9, nesterov = True)
        
        optimizer = Ranger(self.model.parameters(), lr = self.LR, k = 6, N_sma_threshhold = 5, weight_decay = self.WDC)

        return optimizer

    @property
    def optimizer(self):
        return self.freeze_model_bn

    @property
    def choose_scheduler_gen(self):
        scheduler_gen = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max = self.TMAX, eta_min = 0, last_epoch = -1, verbose = True)

        # scheduler_gen = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode = 'min', factor = 0.8, patience = 5, threshold = 0.0001, threshold_mode = 'rel', 
        #     cooldown = 0, min_lr = 0, eps = 1e-08, verbose = 'deprecated')

        return scheduler_gen

    @property
    def scheduler_gen(self):
        return self.choose_scheduler_gen


class ChooseLossFunction:
    ...


class ChooseTransform:
    ...




device = ChooseDevice().device
meta = MetaParameters()
fdwr = FileDirectoryWorker()
chklsz = ChooseKernelSize()

########################################################################################################################
# COMMENTS
########################################################################################################################

try:
    loss_function = nn.CrossEntropyLoss(weight = meta.CE_WEIGHTS).to(device)
    # loss_function = FocalLoss(weight = meta.CE_WEIGHTS).to(device)
    print(f'{loss_function} With {meta.CE_WEIGHTS} Was Chosen !!!')
except:
    # loss_function = nn.CrossEntropyLoss().to(device)
    loss_function = FocalLoss().to(device)
    print(f'{loss_function} without CE_WEIGHTS Was Chosen !!!')


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

