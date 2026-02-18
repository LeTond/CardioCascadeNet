 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.6
Date: 10-02-2026
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""


import torch
from torch.utils.data import DataLoader

import CardioCascadeNet


class TrainRun(CardioCascadeNet.MetaParameters):
    def __init__(self):    
        super(CardioCascadeNet.MetaParameters, self).__init__()

        self.chtfrm = CardioCascadeNet.ChooseTransform()
        self.ds = CardioCascadeNet.DiceLoss()
        self.cmc = CardioCascadeNet.ChooseModelConfig()
        self.jsnlst = CardioCascadeNet.JsonFoldList()
        self.fdwr = CardioCascadeNet.FileDirectoryWorker()
        self.fdwr.create_dir_log(project_name = (self.PROJ_NAME))

        self.jsnlst.create_folds_list
        self.jsnlst.pprint('train_list')
        self.jsnlst.pprint('valid_list')

        self.train_list = self.jsnlst.load_dataset_list('train_list')
        self.valid_list = self.jsnlst.load_dataset_list('valid_list')

        self.plw = CardioCascadeNet.PreprocessLossWeights(self.train_list)
        self.loss_weights_list = self.plw.calculate_loss_weights
        print(self.plw)

        self.model = self.cmc.model
        self.optimizer = self.cmc.optimizer
        self.scheduler_gen = self.cmc.scheduler_gen

        self.transform_01 = self.chtfrm.choose_transforms('transform_01')
        self.transform_02 = self.chtfrm.choose_transforms('transform_02')
        self.transform_03 = self.chtfrm.choose_transforms('transform_03')
        self.transform_04 = self.chtfrm.choose_transforms('transform_04')

    def train_run(self):
        train_ds_images, train_ds_masks, train_ds_templates, train_ds_names = \
                CardioCascadeNet.GetData(self.train_list, self.AUGMENTATION).generated_data_list
        valid_ds_images, valid_ds_masks, valid_ds_templates, valid_ds_names = \
                CardioCascadeNet.GetData(self.valid_list, False).generated_data_list

        loss_function = CardioCascadeNet.ChooseLossFunction(self.loss_weights_list).loss_function

        train_set = \
                CardioCascadeNet.MyDataset(train_ds_images, train_ds_masks, train_ds_templates, train_ds_names, self.transform_01)
        
        for i in range(3):
            train_set += \
                    CardioCascadeNet.MyDataset(train_ds_images, train_ds_masks, train_ds_templates, train_ds_names, self.transform_02)
            train_set += \
                    CardioCascadeNet.MyDataset(train_ds_images, train_ds_masks, train_ds_templates, train_ds_names, self.transform_03)
            train_set += \
                    CardioCascadeNet.MyDataset(train_ds_images, train_ds_masks, train_ds_templates, train_ds_names, self.transform_04)

        train_loader = DataLoader(train_set, self.BT_SZ, drop_last = True, shuffle = True, pin_memory = False)

        valid_set = \
                CardioCascadeNet.MyDataset(valid_ds_images, valid_ds_masks, valid_ds_templates, valid_ds_names, self.transform_01)
        valid_batch_size = len(valid_set)
        valid_loader = \
                DataLoader(valid_set, self.BT_SZ, drop_last = True, shuffle = True, pin_memory = False)

        print(f'\nTrain size: {len(train_set)} | Valid size: {len(valid_set)}\n')
        
        self.model = \
                CardioCascadeNet.TrainNetwork(self.model, self.optimizer, loss_function, self.scheduler_gen, train_loader, valid_loader).train()

    def rewrite_weights_run(self):
        """
        Copy model weights from model to model
        """
        self.cmc.rewrite_weights('ALMAZ', 'HCM_adult', 'Unet1_Fold_full/')





