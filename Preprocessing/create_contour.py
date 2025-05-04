 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.4
Date: 04-05-2025
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""


import nibabel as nib
import numpy as np


import CardioCascadeNet



class NiftiSaver():
    def __init__(self, file_name):         
        self.file_name = file_name

    def save_new_mask(self):
        old_mask = CardioCascadeNet.ReadImages(f'./CardioCascadeNet/Dataset/ALMAZ_mask/{self.file_name}').view_matrix()

        old_mask[old_mask==0] = 20
        old_mask[old_mask==1] = 20
        old_mask[old_mask==2] = 0
        old_mask[old_mask==3] = 0
        old_mask[old_mask==4] = 19

        new_image = nib.Nifti1Image(old_mask, affine = np.eye(4))
        nib.save(new_image, f'./CardioCascadeNet/Dataset/ALMAZ_BULL/{self.file_name}')



class BullEyeContour:
    def __init__(self, path_to_files):
        self.path_to_files = path_to_files

    def get_old_mask(self):
        dataset_list = CardioCascadeNet.ReadImages(self.path_to_files).get_file_path_list()
        return dataset_list

    def save_new_mask(self):
        for sub in self.get_old_mask():
            NiftiSaver(sub).save_new_mask()



if __name__ == "__main__":
    beye = BullEyeContour('./CardioCascadeNet/Dataset/ALMAZ_mask/')
    beye.save_new_mask()





