 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.0
Date: 18-03-2025
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""


import os
import sys
import pydicom as dicom
import nibabel as nib
import numpy as np
 
import matplotlib.pyplot as plt


def get_dcm_name(path_to_file):
    origin_dicom = dicom.dcmread(path_to_file)
    dicom_sub_name = origin_dicom[0x0010, 0x0010].value

    return dicom_sub_name


def get_dcm(path_to_file):
    origin_dicom = dicom.dcmread(path_to_file)
    new_dicom = np.array(origin_dicom.pixel_array)
    
    if len(list(new_dicom.shape)) == 2:
        new_dicom = new_dicom[:, :, np.newaxis]
    else:
        new_dicom = new_dicom.transpose(2, 1, 0)

    return new_dicom


def save_nifti(masks_list, name):
    new_image = nib.Nifti1Image(masks_list, affine = np.eye(4))
    nib.save(new_image, f'./Dataset/testFlip/copy_{name}')


def get_file_list(path_to_file):
    files = os.listdir(path_to_file)
    files.sort()
    return list(files)


class ViewData():
    def view_img(self, img):
        plt.subplot(1, 1, 1)
        plt.imshow(img, cmap = 'gray')
        plt.show()


if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)

    vd = ViewData()

    direct_path = '/Users/aglevchuk/Desktop/Func_DS_CorCTA_15_Qr36_4_10__100__Matrix_256_new_2148/'
    dicoms_list = get_file_list(direct_path)

    for i in range(10):
        dicom_image = get_dcm(direct_path + dicoms_list[200+i*8])

        # image0 = dicom_image[0, 200:344, 200:344]
        # image1 = dicom_image[1, 200:344, 200:344]
        # image2 = dicom_image[2, 200:344, 200:344]

        image0 = dicom_image[0, :, :]
        image1 = dicom_image[1, :, :]
        image2 = dicom_image[2, :, :]

        image = image1 - image0

        # print(image)

        vd.view_img(image)







