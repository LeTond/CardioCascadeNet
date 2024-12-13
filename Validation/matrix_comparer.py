 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.3
Date: 13-12-2024
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""


import os
import numpy as np
from Preprocessing.preprocessing import *



"""
if two images is equal: pass
	else: get names of twiced images in dataset 
"""


def images_list():
	all_images_list = []

	file_list = sorted(os.listdir('./Dataset/HEAD_origin/'))

	for file_name in file_list:
	# for file_name in ['Sub099.nii', 'Sub099.nii']:
		if file_name == '.DS_Store':
			file_list.remove(file_name)
		else:
			image = ReadImages(f"./Dataset/HEAD_origin/{file_name}").view_matrix
			all_images_list.append(np.sum(image))

	return all_images_list, file_list


def delete_repeat_images(images_list_, file_list_):
	copy_images_list_ = images_list_.copy()
	cntr = len(copy_images_list_)

	for indx1, img1 in enumerate(images_list_):
		for indx2, img2 in enumerate(images_list_):
			if np.array_equal(img1, img2) and indx1 != indx2:
				print(sorted(file_list_)[indx1], sorted(file_list_)[indx2])



if __name__ == '__main__':
	images_list, file_list = images_list()
	print(len(images_list))
	print(delete_repeat_images(images_list, file_list))







