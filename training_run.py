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

    def train_run(self):
        model = self.cmc.model
        optimizer = self.cmc.optimizer
        scheduler_gen = self.cmc.scheduler_gen

        transform_01 = self.chtfrm.choose_transforms('transform_01')
        transform_02 = self.chtfrm.choose_transforms('transform_02')
        transform_03 = self.chtfrm.choose_transforms('transform_03')
        transform_04 = self.chtfrm.choose_transforms('transform_04')

        self.jsnlst.create_folds_list

        train_list = self.jsnlst.load_dataset_list('train_list')
        valid_list = self.jsnlst.load_dataset_list('valid_list')

        self.jsnlst.pprint('train_list')
        self.jsnlst.pprint('valid_list')

        train_ds_images, train_ds_masks, train_ds_templates, train_ds_names = CardioCascadeNet.GetData(train_list, self.AUGMENTATION).generated_data_list
        valid_ds_images, valid_ds_masks, valid_ds_templates, valid_ds_names = CardioCascadeNet.GetData(valid_list, False).generated_data_list

        plw = CardioCascadeNet.PreprocessLossWeights(train_list)
        print(plw)
        loss_list = plw.calculate_loss_weights
        loss_function = CardioCascadeNet.ChooseLossFunction(loss_list).loss_function

        train_set = CardioCascadeNet.MyDataset(train_ds_images, train_ds_masks, train_ds_templates, train_ds_names, transform_01)
        for i in range(1):
            train_set += CardioCascadeNet.MyDataset(train_ds_images, train_ds_masks, train_ds_templates, train_ds_names, transform_02)
            train_set += CardioCascadeNet.MyDataset(train_ds_images, train_ds_masks, train_ds_templates, train_ds_names, transform_03)
            train_set += CardioCascadeNet.MyDataset(train_ds_images, train_ds_masks, train_ds_templates, train_ds_names, transform_04)

        train_loader = DataLoader(train_set, self.BT_SZ, drop_last = True, shuffle = True, pin_memory = False)

        valid_set = CardioCascadeNet.MyDataset(valid_ds_images, valid_ds_masks, valid_ds_templates, valid_ds_names, transform_01)
        valid_batch_size = len(valid_set)
        valid_loader = DataLoader(valid_set, self.BT_SZ, drop_last = True, shuffle = True, pin_memory = False)

        print(f'Train size: {len(train_set)} | Valid size: {len(valid_set)}')
        model = CardioCascadeNet.TrainNetwork(model, optimizer, loss_function, scheduler_gen, train_loader, valid_loader).train()

    def rewrite_weights_run(self):
        """
        Copy model weights from model to model
        """
        self.cmc.rewrite_weights('ALMAZ', 'HCM_adult', 'Unet1_Fold_full/')





