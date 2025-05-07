 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.4
Date: 04-05-2025
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""


import sys
import math

import numpy as np
import nibabel as nib
import random as rand
import matplotlib.pyplot as plt

from time import time
from pprint import pprint    


import CardioCascadeNet


def read_nii(path_to_nii):
    # matplotlib.use('TkAgg')
    img = nib.load(path_to_nii)
    return img


def view_matrix(img):
    np.set_printoptions(threshold=sys.maxsize)
    return np.array(img.dataobj)


def view_img(img):
    plt.imshow(img)
    plt.show()


class MaskPostprocessing(CardioCascadeNet.MetaParameters):
    def __init__(self, file_name = None, image = None, masks_list = None, mask_type = None):    
        super(CardioCascadeNet.MetaParameters, self).__init__()
        self.__image = image
        self.__masks_list = masks_list
        self.__mask_type = mask_type
        self.__file_name = file_name

    @property
    def file_name(self):
        return self.__file_name

    @property
    def image(self):
        return self.__image

    @property
    def masks_list(self):
        return self.__masks_list
    
    @property
    def mask_type(self):
        return self.__mask_type

    @staticmethod
    def postprocess_matrix(mask_list):
        shp = list(mask_list[0].shape)
        zero_matrix = np.zeros((len(mask_list), shp[0], shp[1]))

        for slc in range(len(mask_list)):
            zero_matrix[slc, :shp[0], :shp[1]] = mask_list[slc]
        
        mask_list = zero_matrix.copy()
        mask_list = np.array(mask_list, dtype = np.float32)
        mask_list = mask_list.transpose(1, 2, 0)
        mask_list = np.round(mask_list)
        
        return mask_list

    @property
    def check_bull_apex(self):
        masks_list = self.masks_list.copy()
        template = CardioCascadeNet.ReadImages(f"{self.NEW_UNET3_MASKS_PATH}{str(self.file_name).split('/')[-1]}").view_matrix

        shp = list(masks_list.shape)

        zero_matrix = np.zeros((shp[0], shp[1], shp[2]))

        for slc in range(shp[2]):
            new_mask = masks_list[:, :, slc]
            fib_mask = template[:, :, slc]

            if (fib_mask == 1).any():
                zero_matrix[:shp[0], :shp[1], slc] = new_mask
            else:
                fib_mask[fib_mask != 0] = 17
                zero_matrix[:shp[0], :shp[1], slc] = fib_mask

        mask_list = zero_matrix.copy()

        return mask_list


class InstancesFinder():
    def __init__(self, old_matrix, kernel, num_class):
        self.kernel_sz = kernel
        self.old_matrix = old_matrix
        self.extra_symbol = 99
        self.symbols = list(range(2))
        self.num_class = num_class
        self.queue = []
        self.clusters = []
        self.min_distance = 1
        self.min_cluster_size = 1

        # Очередь из символов для поиска кластера
        self.directions_cluster = self.direction_cluster_genertor()

    def direction_cluster_genertor(self):
        directions_cluster = []
        
        for i in range(1, self.min_distance + 1):
            directions_cluster.append([0, i])
            directions_cluster.append([0, -i])
            directions_cluster.append([i, 0])
            directions_cluster.append([-i, 0])
            directions_cluster.append([-i, -i])
            directions_cluster.append([i, i])
            directions_cluster.append([-i, i])
            directions_cluster.append([i, -i])

        return directions_cluster

    def find_clusters(self):
        # Пустая матрица для пометки символов, которые уже участвовали в поиске кластеров
        markedSymbols = [[0 for i in range(self.kernel_sz)] for i in range(self.kernel_sz)] 
        
        # Перебираем все символы матрицы
        for i in range(self.kernel_sz):
            for j in range(self.kernel_sz):
                # Если символ - extra или помечен - пропускаем
                if (self.num_class == self.extra_symbol or markedSymbols[i][j] == 3):
                    continue
                
                clusterData = {
                    'extras': [],
                    'coords': [],
                    'squares': []}
                
                # Добавляем текущий символ в очередь и помечаем его
                self.queue.append([i, j])
                markedSymbols[i][j] = self.num_class

                # Пока в очереди что-то есть - перебираем соседние символы
                while (self.queue):
                    # Забираем символ из очереди
                    coords = self.queue.pop()   
                    # extra и обычные символы добавляем в разные массивы, тк у них разное поведение
                    # print(self.old_matrix[coords[0]][coords[1]], self.extra_symbol)
                    try:
                        if (self.old_matrix[coords[0]][coords[1]] != self.extra_symbol):
                            clusterData['coords'].append(coords)
                        else:
                            clusterData['extras'].append(coords)
                    except:
                        continue

                    # Перебираем все соседние символы
                    for direction in self.directions_cluster:
                        neighbour_coords = [coords[0] + direction[0], coords[1] + direction[1]]
                        try:
                            # Если соседний символ такой же или это extra (и не помечен) - добавляем его в очередь и помечаем
                            if ((self.old_matrix[neighbour_coords[0]][neighbour_coords[1]] == self.num_class or 
                                self.old_matrix[neighbour_coords[0]][neighbour_coords[1]] == self.extra_symbol) and 
                            markedSymbols[neighbour_coords[0]][neighbour_coords[1]] == 0):
                            
                                self.queue.append(neighbour_coords)
                                markedSymbols[neighbour_coords[0]][neighbour_coords[1]] = 3
                        except:
                            pass
                    
                # Берем только те кластеры, у которых длина больше 3 (учитывая extra)
                if (len(clusterData['coords']) + len(clusterData['extras']) >= (self.min_cluster_size + 1)):
                    clusterData['symbol'] = self.num_class
                    self.clusters.append(clusterData)

                # Снимаем пометки с extra текущего кластера, тк они могут быть частью и других кластеров
                for coords in clusterData['extras']:
                    markedSymbols[coords[0]][coords[1]] = 0

        return self.clusters

    def iteration(self):
        ...

    def new_instance_matrix(self):
        """
        Преобразуем обычную 2D маску в instance ndim 
        """
        main_matrix = []
        new_matrix = np.copy(self.new_matrix())

        shp_old = new_matrix.shape

        for clss in np.unique(new_matrix):
            matrix = np.copy(new_matrix)

            if clss < 3:
                matrix[matrix != clss] = 0
                main_matrix.append(matrix)

            elif clss >= 13: 
                matrix[matrix != clss] = 0
                matrix[matrix == clss] = 3
                main_matrix.append(matrix)

        main_matrix = np.array(main_matrix).transpose(2, 1, 0)
        
        shp_new =  main_matrix.shape
        print(f'Matrix shape was changed from {shp_old} to {shp_new}')

        return main_matrix

    def new_matrix(self):
        """
        Для класса self.num_class преобразуем каждый отдельный кластер в новый инстанс 
        """
        new_matrix = np.copy(self.old_matrix)
        cluster = self.find_clusters()

        for i in range(len(cluster[:])):
            for j in range(1, len(cluster[i]['coords'])):

                new_layer = self.num_class + i + 10 # i - from 0 to count of found classes
                coord_ = cluster[i]['coords'][j]
                new_matrix[coord_[0]][coord_[1]] = new_layer
                
        return new_matrix

    def threshold_matrix(self):
        """
        Для группы пикселей размером <= 3 - назначаем класс 99
        """
        new_matrix = np.copy(self.old_matrix)
        cluster = self.find_clusters()

        for i in range(len(cluster[:])):
            cluster_size = len(cluster[i]['coords'])
            print(f' Размер {i + 1}-го кластера в пикселях {cluster_size - 1} ')
            
            if len(cluster[i]['coords']) <= 3:
                for j in range(1, len(cluster[i]['coords'])):
                    new_layer = 99 # i - from 0 to count of found classes
                    coord_ = cluster[i]['coords'][j]
                    new_matrix[coord_[0]][coord_[1]] = new_layer
                    
        return new_matrix

    def transcheck(self):
        clusters = self.find_clusters()
        answer_list = []

        cluster_size = len(clusters)
        
        if cluster_size > 0:
            for ilnd in range(cluster_size):
                neighbor_list = []

                for clms, row in clusters[ilnd]['coords']:
                    clms += 1
                    row += 1
                    neighbor_list.append(self.old_matrix[clms + 1][row + 1])
                    neighbor_list.append(self.old_matrix[clms + 0][row + 1])
                    neighbor_list.append(self.old_matrix[clms + 1][row + 0])

                    neighbor_list.append(self.old_matrix[clms - 1][row - 1])
                    neighbor_list.append(self.old_matrix[clms - 0][row - 1])
                    neighbor_list.append(self.old_matrix[clms - 1][row - 0])

                    neighbor_list.append(self.old_matrix[clms + 1][row - 1])
                    neighbor_list.append(self.old_matrix[clms - 1][row + 1])

                if '0' in str(set(neighbor_list)) and '1' in str(set(neighbor_list)):
                    answer_list.append('3')
                elif '0' in str(set(neighbor_list)) and '1' not in str(set(neighbor_list)):
                    answer_list.append('2')
                elif '1' in str(set(neighbor_list)) and '0' not in str(set(neighbor_list)):
                    answer_list.append('1')
                elif '1' not in str(set(neighbor_list)) and '0' not in str(set(neighbor_list)):
                    answer_list.append('4')
                else:
                    answer_list.append('0')
        else:
            answer_list.append('-')
        
        return answer_list


if __name__ == "__main__":
    old_matrix = [
                [0, 3, 1, 0, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 3, 3, 3, 0, 1],
                [1, 0, 1, 3, 3, 3, 1, 0, 1],
                [1, 1, 3, 0, 1, 1, 0, 1, 3],
                [1, 1, 0, 1, 3, 3, 3, 1, 3],
                [1, 1, 0, 1, 1, 3, 3, 1, 0],
                [0, 3, 1, 0, 1, 0, 0, 0, 1],
                [0, 3, 3, 2, 2, 0, 3, 1, 1],
                [0, 0, 2, 2, 2, 1, 1, 1, 0]
                ]

    # generate it to npy

    image_matrix = view_matrix(read_nii("./Sub03.nii"))
    image_matrix = image_matrix[:, :, 4]
    # new_instance_matrix = InstancesFinder(image_matrix, kernel = 144).new_instance_matrix()
    # new_instance_matrix = InstancesFinder(image_matrix, kernel = 144).new_matrix()
    
    new_matrix = InstancesFinder(old_matrix, kernel = 9).new_matrix()
    print(new_matrix)
    # view_img(new_instance_matrix)


    threshold_matrix = InstancesFinder(old_matrix, kernel = 9).threshold_matrix()
    print(threshold_matrix)



    # fromn  (4, 144, 192) to  (144, 192, 4)




