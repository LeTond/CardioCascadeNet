 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.4
Date: 04-05-2025
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""


import os

import numpy as np
import nibabel as nib
 


def get_nii(path_to_file):
    img = nib.load(path_to_file)

    return img


def view_matrix(path):
    matrix = np.array(get_nii(path).dataobj)

    return matrix


def save_nifti(masks_list, name):
    new_image = nib.Nifti1Image(masks_list, affine = np.eye(4))
    nib.save(new_image, f'./CardioCascadeNet/Dataset/testFlip/copy_{name}')



if __name__ == '__main__':
    for name in os.listdir('./CardioCascadeNet/Dataset/testFlip'):
        try:
            masks = view_matrix(f'./CardioCascadeNet/Dataset/testFlip/{name}')
            
            new_masks = []       
            slc_list = []

            for slc in range(masks.shape[2]):
                if np.max(masks[:, :, -slc-1]) == 0:
                    slc_list.append(0)
                else:
                    slc_list.append(1)

            print(f'name: {name}, length: {masks.shape[2]}, slc_list: {slc_list[::-1]}')

            for slc in range(masks.shape[2]):
                mask = masks[:, :, -slc-1]

                # mask = np.fliplr(mask)    # зеркалим вместе с transpose(1, 2, 0)

                new_masks.append(mask)

            new_masks = np.array(new_masks, dtype = np.float32)
            # new_masks = new_masks.transpose(2, 1, 0)

            new_masks = new_masks.transpose(1, 2, 0)    # второй круг - без fliplr


            save_nifti(new_masks, name)
        except:
            pass







