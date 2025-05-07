 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.4
Date: 04-05-2025
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""



import torch
from torchsummary import summary
from torch.utils.data import DataLoader

import CardioCascadeNet


class TrainRun(CardioCascadeNet.MetaParameters):
    def __init__(self):    
        super(CardioCascadeNet.MetaParameters, self).__init__()

        self.chtfrm = CardioCascadeNet.ChooseTransform()
        self.ds = CardioCascadeNet.DiceLoss()
        self.cmc = CardioCascadeNet.ChooseModelConfig()
        self.jsnlst = CardioCascadeNet.JsonFoldList()
        self.loss_function = CardioCascadeNet.ChooseLossFunction().loss_function
        CardioCascadeNet.FileDirectoryWorker().create_dir_log(project_name = self.PROJ_NAME)

    def train_run(self):
        model = self.cmc.model
        optimizer = self.cmc.optimizer
        scheduler_gen = self.cmc.scheduler_gen

        # summary(model, input_size = (1, 64, 64))  # didn't work on ARM GPU

        transform_01 = self.chtfrm.choose_transforms('transform_01')
        transform_02 = self.chtfrm.choose_transforms('transform_02')
        transform_03 = self.chtfrm.choose_transforms('transform_03')
        transform_05 = self.chtfrm.choose_transforms('transform_05')
        transform_06 = self.chtfrm.choose_transforms('transform_06')

        self.jsnlst.create_folds_list

        train_list = self.jsnlst.load_dataset_list('train_list')
        valid_list = self.jsnlst.load_dataset_list('valid_list')

        self.jsnlst.pprint('train_list')
        self.jsnlst.pprint('valid_list')

        train_ds_images, train_ds_masks, train_ds_templates, train_ds_names = CardioCascadeNet.GetData(train_list, self.AUGMENTATION).generated_data_list
        valid_ds_images, valid_ds_masks, valid_ds_templates, valid_ds_names = CardioCascadeNet.GetData(valid_list, False).generated_data_list

        train_set = CardioCascadeNet.MyDataset(train_ds_images, train_ds_masks, train_ds_templates, train_ds_names, transform_01)
        for i in range(3):
            train_set += CardioCascadeNet.MyDataset(train_ds_images, train_ds_masks, train_ds_templates, train_ds_names, transform_02)
            train_set += CardioCascadeNet.MyDataset(train_ds_images, train_ds_masks, train_ds_templates, train_ds_names, transform_03)
            # train_set += CardioCascadeNet.MyDataset(train_ds_images, train_ds_masks, train_ds_templates, train_ds_names, transform_05)
            # train_set += CardioCascadeNet.MyDataset(train_ds_images, train_ds_masks, train_ds_templates, train_ds_names, transform_06)

        train_loader = DataLoader(train_set, self.BT_SZ, drop_last = True, shuffle = True, pin_memory = False)

        valid_set = CardioCascadeNet.MyDataset(valid_ds_images, valid_ds_masks, valid_ds_templates, valid_ds_names, transform_01)
        valid_batch_size = len(valid_set)
        valid_loader = DataLoader(valid_set, self.BT_SZ, drop_last = True, shuffle = True, pin_memory = False)

        print(f'Train size: {len(train_set)} | Valid size: {len(valid_set)}')
        model = CardioCascadeNet.TrainNetwork(model, optimizer, self.loss_function, scheduler_gen, train_loader, valid_loader).train()

    def rewrite_weights_run(self):
        ########################################################################################################################
        # Creating loaders for training and validating network
        ########################################################################################################################
        # self.cmc.rewrite_weights('ALMAZ', 'HCM_adult', 'Unet1_Fold_full/')
        # self.cmc.rewrite_weights('ALMAZ', 'HCM_adult', 'Unet2_Fold_full/')
        # self.cmc.rewrite_weights('ALMAZ', 'HCM_adult', 'Unet3_Fold_full/')
        # self.cmc.rewrite_weights('ALMAZ', 'HCM_adult', 'Unet4_Fold_full/')
        # self.cmc.rewrite_weights('ALMAZ', 'HCM_adult', 'Unet5_Fold_full/')
        # self.cmc.rewrite_weights('BULLEYE', 'ALMAZ', 'Unet5_Fold_full/')
        # self.cmc.rewrite_weights('BULLEYE', 'ALMAZ', 'Unet4_Fold_full/')
        # self.cmc.rewrite_weights('HCM_adult', 'HCM_adult', 'Unet2_Fold_full/')
        # self.cmc.rewrite_weights('HCM_adult', 'Full', 'Unet5_Fold_full/')
        ...


if __name__ == '__main__':
    TrainRun().rewrite_weights_run()
    TrainRun().train_run()





