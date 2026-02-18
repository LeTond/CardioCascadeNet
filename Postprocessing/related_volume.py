 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.6
Date: 10-02-2026
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""

import os
import numpy as np
import nibabel as nib


import CardioCascadeNet



class CountRelVolume(CardioCascadeNet.MetaParameters):
    def __init__(self, path_to_fibmask: str, path_to_bullmasks: str):
        super(CardioCascadeNet.MetaParameters, self).__init__()

        self.path_to_fibmask = path_to_fibmask
        self.path_to_bullmasks = path_to_bullmasks

        self.fibrosis = self.load_matrix(self.path_to_fibmask, False)
        self.bulleye = self.load_matrix(self.path_to_bullmasks, True)
        self.length = self.bulleye.shape[-1]

        self.smooth = 1e-5

    def sub_name(self):
        name = self.path_to_fibmask.split('/')[-1]
        name = name.rstrip('.nii')
        
        return name

    @staticmethod
    def load_matrix(path_to_matrix:str, cond:bool):
        matrix = nib.load(path_to_matrix)
        matrix = np.array(matrix.dataobj)

        if np.max(matrix) > 17: 
            print("ERRRROOOR")

        return matrix

    def rel_volume(self):
        num_class = len(self.DICT_CLASS)
        dictionary = {}

        for key in range(1, num_class):
            dictionary[f'rVlm_{self.DICT_CLASS[key]}'] = 0

        for slc in range(self.length):
            fibrosis = self.fibrosis[:,:,:]
            bulleye = self.bulleye[:,:,:]
            
        for key in range(1, num_class):
            fibrosis_ = fibrosis.copy()
            bulleye_ = bulleye.copy()

            bulleye_[bulleye_ != key] = 0
            bulleye_[bulleye_ == key] = 1
            fibrosis_ = fibrosis_ * bulleye_

            fib = fibrosis_[fibrosis_ == 3]
            myo = fibrosis_[fibrosis_ == 2]

            fib[fib!=3] = 0
            fib[fib!=0] = 1

            myo[myo!=2] = 0
            myo[myo!=0] = 1

            fib = fib.sum().item()
            myo = myo.sum().item()

            rel_volume = int((fib) / (fib + myo + self.smooth) * 100)
            dictionary[f'rVlm_{self.DICT_CLASS[key]}'] = rel_volume 
            
        return dictionary

    def rel_aortic_volume(self):
        num_class = len(self.DICT_CLASS)
        dictionary = {}

        for key in range(1, num_class):
            dictionary[f'rVlm_{self.DICT_CLASS[key]}'] = 0

        for slc in range(self.length):
            fibrosis = self.fibrosis[:,:,:]
            bulleye = self.bulleye[:,:,:]

        bulleye[bulleye == 1] = 1
        bulleye[bulleye == 2] = 1
        bulleye[bulleye == 3] = 2
        bulleye[bulleye == 4] = 2
        bulleye[bulleye == 5] = 3
        bulleye[bulleye == 6] = 3
        bulleye[bulleye == 7] = 1
        bulleye[bulleye == 8] = 1
        bulleye[bulleye == 9] = 2
        bulleye[bulleye == 10] = 2
        bulleye[bulleye == 11] = 3
        bulleye[bulleye == 12] = 3
        bulleye[bulleye == 13] = 1
        bulleye[bulleye == 14] = 1
        bulleye[bulleye == 15] = 2
        bulleye[bulleye == 16] = 3
        bulleye[bulleye == 17] = 4

        # bulleye[bulleye != 1] = 1

        for key in range(1, num_class):
            fibrosis_ = fibrosis.copy()
            bulleye_ = bulleye.copy()

            bulleye_[bulleye_ != key] = 0
            bulleye_[bulleye_ == key] = 1

            fibrosis_ = fibrosis_ * bulleye_

            fib = fibrosis_[fibrosis_ == 3]
            myo = fibrosis_[fibrosis_ == 2]

            fib[fib!=3] = 0
            fib[fib!=0] = 1

            myo[myo!=2] = 0
            myo[myo!=0] = 1

            fib = fib.sum().item()
            myo = myo.sum().item()

            rel_volume = int((fib) / (fib + myo + self.smooth) * 100)
            dictionary[f'rVlm_{self.DICT_CLASS[key]}'] = rel_volume
        
        return dictionary

    def check_transmural(self):
        """
        Example: Sub22
        :return: {'Seg 01': ['-'], 'Seg 02': ['3'], 'Seg 03': ['3'], 'Seg 04': ['3'], 'Seg 05': ['3'], 'Seg 06': ['-'], 
        'Seg 07': ['2', '3'], 'Seg 08': ['2', '3'], 'Seg 09': ['3'], 'Seg 10': ['-'], 'Seg 11': ['3'], 'Seg 12': ['2', '3'], 
        'Seg 13': ['3'], 'Seg 14': ['3'], 'Seg 15': ['3'], 'Seg 16': ['3'], 
        'Seg 17': ['-']}
        Where are:  '-' - segment without fibrosis; 
                    '0' - None
                    '1' - subEndo fibrosis
                    '2' - subEpi fibrosis
                    '3' - Transmural fibrosis
                    '4' - Intramural fibrosis
        """

        num_class = len(self.DICT_CLASS)
        dictionary = {}

        for key in range(1, num_class):
            dictionary[f'{self.DICT_CLASS[key]}'] = []

        for slc in range(self.length):
            fibrosis = self.fibrosis[:, :, slc]
            bulleye = self.bulleye[:, :, slc]
        
            for key in range(1, num_class):
                fibrosis1 = fibrosis.copy()
                fibrosis2 = fibrosis.copy()
                
                bulleye1 = bulleye.copy()
                bulleye2 = bulleye.copy()
                
                bulleye1[bulleye1 != key] = 0
                bulleye1[bulleye1 == key] = 1

                bulleye2[bulleye2 != key] = 1
                bulleye2[bulleye2 == key] = 0

                fibrosis_new2 = fibrosis2 * bulleye2
                fibrosis_new2[fibrosis_new2 == 2] = 97
                fibrosis_new2[fibrosis_new2 == 3] = 98

                fibrosis_new1 = fibrosis1 * bulleye1
                fibrosis_new = fibrosis_new1 + fibrosis_new2

                seg_check = CardioCascadeNet.InstancesFinder(fibrosis_new, kernel = 144, num_class = 3).transcheck()
                dictionary[f'{self.DICT_CLASS[key]}'] += ([i for i in seg_check])
        
        for key in range(1, num_class):
            dictionary[f'{self.DICT_CLASS[key]}'] = list(set(dictionary[f'{self.DICT_CLASS[key]}']))
            if len(dictionary[f'{self.DICT_CLASS[key]}']) > 1:
                dictionary[f'{self.DICT_CLASS[key]}'].remove('-')

        return dictionary

    def print(self):
        print(self.sub_name())
        print(self.rel_volume())
        # print(self.rel_aortic_volume())
        # print(self.check_transmural())

    def get_text(self):
        text = ''
        diction_vol = self.rel_volume()

        for key in diction_vol.keys():
            text += f"{key}: {diction_vol[key]} % \n"

        # print(self.rel_aortic_volume())
        # print(self.check_transmural())
        return text


class CountRelvolumeRun(CardioCascadeNet.MetaParameters):
    def __init__(self):
        super().__init__()
        super(CardioCascadeNet.MetaParameters, self).__init__()

    def count_rel_volume_run(self):
        _root_directory_path = os.path.abspath(os.path.dirname('..'))

        jsnlst = CardioCascadeNet.JsonFoldList()
        jsnlst.create_folds_list
        dataset_list = jsnlst.load_dataset_list('test_list')
        print(dataset_list)

        for sub_name in dataset_list:
            path_to_fibmask = f'{self.NEW_UNET2_MASKS_PATH}{sub_name}'
            path_to_bullmask = f'{self.NEW_UNET5_MASKS_PATH}{sub_name}'

            cm = CountRelVolume(path_to_fibmask, path_to_bullmask)
            cm.print()

