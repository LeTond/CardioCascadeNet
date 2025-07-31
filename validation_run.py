 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.4
Date: 04-05-2025
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""

import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import ptitprince as pt
# import seaborn as sns


from torch.utils.data import DataLoader


import CardioCascadeNet


class PlotResults(CardioCascadeNet.MetaParameters):
    def __init__(self):         
        super(CardioCascadeNet.MetaParameters, self).__init__()

        if self.UNET2 is True:
            self.kernel_sz = self.CROPP_KERNEL
        elif self.UNET2 is False:
            self.kernel_sz = self.KERNEL

        self.dict_class_stats = self.create_dict_class()
        self.default_transform = CardioCascadeNet.ChooseTransform().choose_transforms('transform_01')

    def data_loader(self, data_list, augmentation = False):
        getds_origin, getds_mask, getds_template, getds_names = CardioCascadeNet.GetData(data_list, augmentation).generated_data_list
        data_set = CardioCascadeNet.MyDataset(getds_origin, getds_mask, getds_template, getds_names, self.default_transform)
        
        data_batch_size = len(data_set)
        data_loader = DataLoader(data_set, data_batch_size, drop_last = True, shuffle = False, pin_memory = True)

        return data_loader

    def create_dict_class(self):
        dict_class_stats = {}

        for key in range(1, self.NUM_CLASS): 
            dict_class_stats[f'Precision_{self.DICT_CLASS[key]}'] = []
            dict_class_stats[f'Recall_{self.DICT_CLASS[key]}'] = []
            dict_class_stats[f'Accuracy_{self.DICT_CLASS[key]}'] = []
            dict_class_stats[f'Dice_{self.DICT_CLASS[key]}'] = []

            dict_class_stats[f'FN_{self.DICT_CLASS[key]}'] = []
            dict_class_stats[f'FP_{self.DICT_CLASS[key]}'] = []
            dict_class_stats[f'GTPix_{self.DICT_CLASS[key]}'] = []
            dict_class_stats[f'CMPix_{self.DICT_CLASS[key]}'] = []

            dict_class_stats[f'GTVol_{self.DICT_CLASS[key]}'] = []
            dict_class_stats[f'CMVol_{self.DICT_CLASS[key]}'] = []

        return dict_class_stats

    def bland_altman_per_subject(self, model, test_list):
        for subj in test_list:
            try:
                for key in range(1, self.NUM_CLASS): 
                    data_loader = self.data_loader([subj], False)
                    tm = CardioCascadeNet.TissueMetrics(model, data_loader)
                    dict_class_sub_stats = tm.bland_altman_metrics()
                    
                    self.dict_class_stats[f'GTVol_{self.DICT_CLASS[key]}'].append(np.sum(dict_class_sub_stats[f'GTVol_{self.DICT_CLASS[key]}']))
                    self.dict_class_stats[f'CMVol_{self.DICT_CLASS[key]}'].append(np.sum(dict_class_sub_stats[f'CMVol_{self.DICT_CLASS[key]}']))
            
            except ValueError:
                print(f'Subject {subj} has no suitable images !!!!')

        return self.dict_class_stats

    def stats_per_subject(self, model, test_list):
        for subj in test_list:
            try:
                data_loader = self.data_loader([subj], False)
                tm = CardioCascadeNet.TissueMetrics(model, data_loader)
                dict_class_sub_stats = tm.image_metrics()

                for key in range(1, self.NUM_CLASS):
                    self.dict_class_stats[f'Precision_{self.DICT_CLASS[key]}'] += dict_class_sub_stats[f'Precision_{self.DICT_CLASS[key]}']
                    self.dict_class_stats[f'Recall_{self.DICT_CLASS[key]}'] += dict_class_sub_stats[f'Recall_{self.DICT_CLASS[key]}']
                    self.dict_class_stats[f'Accuracy_{self.DICT_CLASS[key]}'] += dict_class_sub_stats[f'Accuracy_{self.DICT_CLASS[key]}']
                    self.dict_class_stats[f'Dice_{self.DICT_CLASS[key]}'] += dict_class_sub_stats[f'Dice_{self.DICT_CLASS[key]}']

            except ValueError:
                print(f'Subject {subj} has no suitable images !!!!')

        return self.dict_class_stats

    def prepare_plot(self, sub_names, origImage, origMask, predMask, dice_layers):
        figure, ax = plt.subplots(nrows = 1, ncols = 4, figsize = (12, 12))

        origImage = np.resize(origImage.cpu(), (self.kernel_sz, self.kernel_sz))        
        predMask = np.resize(predMask.cpu(), (self.kernel_sz, self.kernel_sz))
        origMask = np.resize(origMask.cpu(), (self.kernel_sz, self.kernel_sz))

        for key in range(1, self.NUM_CLASS):
            predMask[0][key-1] = key
            origMask[0][key-1] = key

        colormap = plt.cm.get_cmap('viridis')  # 'plasma' or 'viridis'
        colormap.set_under('k', alpha=0.5)

        ax[0].imshow(origImage, plt.get_cmap('gray'))
        ax[1].imshow(origImage, plt.get_cmap('gray'))
        ax[1].imshow(origMask, alpha = 0.5, interpolation=None, cmap=colormap,  vmin=0.5)
        ax[1].contour(origMask, alpha = 0.5)

        ax[2].imshow(origImage, plt.get_cmap('gray'))
        ax[2].imshow(predMask, alpha = 0.5, interpolation=None, cmap=colormap,  vmin=0.5)
        ax[2].contour(predMask, alpha = 0.5)
        ax[3].imshow(predMask, alpha = 0.5)

        ax[0].set_title(f"{sub_names}", fontsize = 10, fontweight = 'bold')
        ax[1].set_title(f"Dice: {dice_layers} \nManual mask", fontsize = 10, fontweight ='bold')
        ax[2].set_title(f"Computed mask", fontsize = 10, fontweight='bold')
        ax[3].set_title(f"Computed mask", fontsize = 10, fontweight='bold')
        
        figure.set_edgecolor("green")
        figure.tight_layout()
        
        return figure

    def show_predicted(self, predicted_masks):
        for i in range(predicted_masks[0][0]):
            dice_layers = str('')

            for key in range(1, self.NUM_CLASS):
                # if round(predicted_masks[5].get(f'{self.DICT_CLASS[key]}')[i], 3) > 0 and round(predicted_masks[5].get(f'{self.DICT_CLASS[key]}')[i], 3) != 1:
                dice_layers += f' {self.DICT_CLASS[key]} = '
                dice_layers += str(round(predicted_masks[5].get(f'{self.DICT_CLASS[key]}')[i], 3))

            # if round(predicted_masks[5].get(f'{self.DICT_CLASS[key]}')[i], 3) == 0.0:
            self.prepare_plot(predicted_masks[1][i], predicted_masks[2][i], predicted_masks[3][i], predicted_masks[4][i], dice_layers)

    def create_hist(self, value_list: list):
        img_np = np.array(value_list)
        plt.hist(img_np.ravel(), bins=20, density=False)
        plt.xlabel("DSC")
        plt.ylabel("Images")
        plt.title("Distribution of dice")


class ValidationRun(CardioCascadeNet.MetaParameters):
    def __init__(self):         
        super(CardioCascadeNet.MetaParameters, self).__init__()

        self.pltres = PlotResults()
        self.jsnlst = CardioCascadeNet.JsonFoldList()
        self.ds = CardioCascadeNet.DiceLoss()
        self.neural_model = self.choose_model
        self.kernel_sz = self.choose_kernel_sz

    @property
    def choose_model(self):
        checkpoint = torch.load(f'{self.PROJ_NAME}/{self.DATASET_NAME}_model.pth', weights_only=False)
       
        if self.UNET2 is False and self.UNET3 is False:
            checkpoint = checkpoint[f'Net_{self.DATASET_NAME}_{self.UNET1_FOLD}']
            neural_model = checkpoint[f'Model']
            neural_model.load_state_dict(checkpoint['weights'])
       
        elif self.UNET2 is True and self.UNET3 is False:
            checkpoint = checkpoint[f'Net_{self.DATASET_NAME}_{self.UNET2_FOLD}']
            neural_model = checkpoint[f'Model']
            neural_model.load_state_dict(checkpoint['weights'])
         
        elif self.UNET3 is True and self.UNET4 is False:
            checkpoint = checkpoint[f'Net_{self.DATASET_NAME}_{self.UNET3_FOLD}']
            neural_model = checkpoint[f'Model']
            neural_model.load_state_dict(checkpoint['weights'])
         
        elif self.UNET4 is True and self.UNET5 is False:
            checkpoint = checkpoint[f'Net_{self.DATASET_NAME}_{self.UNET4_FOLD}']
            neural_model = checkpoint[f'Model']
            neural_model.load_state_dict(checkpoint['weights'])
           
        elif self.UNET5 is True:
            checkpoint = checkpoint[f'Net_{self.DATASET_NAME}_{self.UNET5_FOLD}']
            neural_model = checkpoint[f'Model']
            neural_model.load_state_dict(checkpoint['weights'])
          
        return neural_model

    @property
    def choose_kernel_sz(self):
        if self.UNET2 is False and self.UNET3 is False:
            kernel_sz = self.KERNEL
        elif self.UNET2 is True and self.UNET3 is False:
            kernel_sz = self.CROPP_KERNEL 
        elif self.UNET3 is True and self.UNET4 is False:
            kernel_sz = self.CROPP_KERNEL 
        elif self.UNET4 is True and self.UNET5 is False:
            kernel_sz = self.CROPP_KERNEL 
        elif self.UNET5 is True:
            kernel_sz = self.CROPP_KERNEL 

        return kernel_sz

    def show_dict_class_stats(self, dict_class_stats):
        for key in range(1, self.NUM_CLASS):
            print('')
            print(f'Class_{self.DICT_CLASS[key]}')
            
            print(
                f'DSC: '
                f'Mean - {round(np.mean(dict_class_stats[f"Dice_{self.DICT_CLASS[key]}"]), 3)} '
                f'Median - {round(np.median(dict_class_stats[f"Dice_{self.DICT_CLASS[key]}"]), 3)} '
                )
            
            print(
                f'Precision: '
                f'Mean - {round(np.mean(dict_class_stats[f"Precision_{self.DICT_CLASS[key]}"]), 3)} '
                f'Median - {round(np.median(dict_class_stats[f"Precision_{self.DICT_CLASS[key]}"]), 3)} '
                )
            
            print(
                f'Recall: '
                f'Mean - {round(np.mean(dict_class_stats[f"Recall_{self.DICT_CLASS[key]}"]), 3)} '
                f'Median - {round(np.median(dict_class_stats[f"Recall_{self.DICT_CLASS[key]}"]), 3)} '
                )

        for key in range(1, self.NUM_CLASS):
            self.pltres.create_hist(dict_class_stats[f"Dice_{self.DICT_CLASS[key]}"])

    def validation_run(self):
        test_list = self.jsnlst.load_dataset_list('test_list')

        for subject in test_list:
            test_loader = self.pltres.data_loader([subject], False)

            print(f'Test size: {len(test_list)}')

            show_predicted_masks = CardioCascadeNet.MaskPrediction().prediction_masks(self.neural_model, test_loader)
            self.pltres.show_predicted(show_predicted_masks)

            # Vgt, Vcm = CardioCascadeNet.TissueMetrics(self.neural_model, test_loader).relative_volume()
            # print(Vgt, Vcm)

    @property
    def get_dict_class_stats(self):
        test_list = self.jsnlst.load_dataset_list('test_list')

        return self.pltres.stats_per_subject(self.neural_model, test_list)

    def show_hist_run(self):
        try:
            self.show_dict_class_stats(self.get_dict_class_stats)

        except ValueError:
            print(f'Subjects has no suitable images !!!!')

    def rain_cloud(self, dict_class_stats):
        dsc_list = []

        for key in range(1, self.NUM_CLASS):
            dsc_list += dict_class_stats[f"Dice_{self.DICT_CLASS[key]}"]

        tissue_list = ['ЛЖ' for i in range(len(dsc_list))] + ['Миокард' for i in range(len(dsc_list))] + ['Фиброз' for i in range(len(dsc_list))]

        df = pd.DataFrame(
            {
            'Структура': tissue_list,
            'Метрика DSC': dsc_list,
            }
        )

        f, ax = plt.subplots(figsize = (10, 5))

        # ax = pt.RainCloud(
        #     x = 'Tissues', y = 'DSC', data = df,
        #     pointplot = True, width_viol = .8, width_box = .8, linewidth = 1, alpha = 0.8, bw = 0.1, scale = "area", orient = 'h', move = .0,
        # )

        ax = pt.RainCloud(
            x = 'Структура', y = 'Метрика DSC', data = df,
            pointplot = False, width_viol = 0.8, width_box = 0.8, linewidth = 1, alpha = 0.8, bw = 0.1, scale = "area", orient = 'h', move = 0.0,
        )

        for key in range(1, self.NUM_CLASS):
            print(
                f'{self.DICT_CLASS[key]} '
                f'{pd.DataFrame(dict_class_stats[f"Dice_{self.DICT_CLASS[key]}"]).quantile([0.25, 0.5, 0.75])} '
        )

    def rain_cloud_run(self):        
        try:
            test_list = self.jsnlst.load_dataset_list('test_list')
            dict_class_stats = self.pltres.stats_per_subject(self.neural_model, test_list)
            # self.rain_cloud(dict_class_stats)

        except ValueError:
            print(f'Subjects has no suitable images !!!!')

    @property
    def get_bland_altman_per_subject(self):
        test_list = self.jsnlst.load_dataset_list('test_list')
        
        try:
            return self.pltres.bland_altman_per_subject(self.neural_model, test_list)
        
        except ValueError:
            print(f'Subjects has no suitable images !!!!')



if __name__ == "__main__":
    ValidationRun().validation_run()
