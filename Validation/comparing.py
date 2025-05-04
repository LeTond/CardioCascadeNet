 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.4
Date: 04-05-2025
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""


import numpy as np
import pandas as pd
import nibabel as nib

from medpy import metric
from scipy.ndimage import _ni_support
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure


import CardioCascadeNet



class CompareMatrix(CardioCascadeNet.MetaParameters):
    def __init__(self, path_to_mask: str, path_to_prediction: str, num_class: int):
        super(CardioCascadeNet.MetaParameters, self).__init__()

        self.path_to_mask = path_to_mask
        self.path_to_prediction = path_to_prediction
        self.num_class = num_class

        self.mask = self.load_matrix(self.path_to_mask, num_class, False, 'etalon')
        self.prediction = self.load_matrix(self.path_to_prediction, num_class, True, 'predict')
        
        self.length = self.mask.shape[-1]

        self.smooth = 1e-5
        self.GT = self.mask.sum()
        self.CM = self.prediction.sum()
        self.TP = (self.mask * self.prediction).sum()
        self.FN = np.abs(self.GT - self.TP)
        self.FP = np.abs(self.CM - self.TP)

    def sub_name(self):
        name = self.path_to_mask.split('/')[-1]
        name = name.rstrip('.nii')
        
        return name

    @staticmethod
    def load_matrix(path_to_matrix: str, num_class: int, cond: bool, marker: str):
        matrix = nib.load(path_to_matrix)
        matrix = np.array(matrix.dataobj)

        # if cond is True:
        # for slc in range(matrix.shape[-1]):        
        #     if matrix[:,:,slc][matrix[:,:,slc]==3].sum().item() < 10:
        #         matrix[:,:,slc][matrix[:,:,slc]==3] = 2

          # Myo + Fib
        # if num_class == 2:
        #     matrix[matrix==3] = 2

        matrix[matrix != num_class] = 0
        matrix[matrix == num_class] = 1

        return matrix                                     

    def dice_2d(self):
        list_dsc = []
        for slc in range(self.length):
            self.GT = self.mask[:, :, slc].sum()
            self.CM = self.prediction[:, :, slc].sum()

            if self.GT == 0:
                pass
            else:
                self.TP = (self.mask[:, :, slc] * self.prediction[:, :, slc]).sum()
                self.FN = np.abs(self.GT - self.TP)
                self.FP = np.abs(self.CM - self.TP)
    
                # print(f'{self.sub_name()}: FN pixels {self.FN}, FP pixels {self.FP}, TP pixels: {self.TP}')
                # print(f'{self.sub_name()} Slice: {self.length - slc} Dice = {self.dice()}')
                list_dsc.append(self.dice())

        print(f"Sub {self.sub_name()}: {list_dsc}")

        return list_dsc

    def recall_2d(self):
        list_rcl = []

        for slc in range(self.length):
            self.GT = self.mask[:, :, slc].sum()
            self.CM = self.prediction[:, :, slc].sum()
            self.TP = (self.mask[:, :, slc] * self.prediction[:, :, slc]).sum()
            self.FN = np.abs(self.GT - self.TP)
            self.FP = np.abs(self.CM - self.TP)
            
            # print(f'{self.sub_name()} Slice: {self.length - slc} Recall = {self.recall()}')
            list_rcl.append(self.recall())
        
        # print(f"Sub {self.sub_name()}: {list_rcl}")

        return list_rcl

    def precision_2d(self):
        list_prsn = []

        for slc in range(self.length):
            self.GT = self.mask[:, :, slc].sum()
            self.CM = self.prediction[:, :, slc].sum()
            self.TP = (self.mask[:, :, slc] * self.prediction[:, :, slc]).sum()
            self.FN = np.abs(self.GT - self.TP)
            self.FP = np.abs(self.CM - self.TP)
            
            # print(f'{self.sub_name()} Slice: {self.length - slc} Precision = {self.precision()}')
            list_prsn.append(self.precision())
        
        # print(f"Sub {self.sub_name()}: {list_prsn}")

        return list_prsn

    @staticmethod
    def surface_distances(result, reference, voxelspacing = None, connectivity = 1):
        """
        The distances between the surface voxel of binary objects in result and their
        nearest partner surface voxel of a binary object in reference.
        """
        result = np.atleast_1d(result.astype(np.bool_))
        reference = np.atleast_1d(reference.astype(np.bool_))
        if voxelspacing is not None:
            voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
            voxelspacing = np.asarray(voxelspacing, dtype = np.float64)
            if not voxelspacing.flags.contiguous:
                voxelspacing = voxelspacing.copy()
                
        # binary structure
        footprint = generate_binary_structure(result.ndim, connectivity)
        
        # test for emptiness
        if 0 == np.count_nonzero(result): 
            raise RuntimeError('The first supplied array does not contain any binary object.')
        if 0 == np.count_nonzero(reference): 
            raise RuntimeError('The second supplied array does not contain any binary object.')    
                
        # extract only 1-pixel border line of objects
        result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
        reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
        
        # compute average surface distance        
        # Note: scipys distance transform is calculated only inside the borders of the
        #       foreground objects, therefore the input has to be reversed
        dt = distance_transform_edt(~reference_border, sampling = voxelspacing)
        sds = dt[result_border]
        
        return sds

    def hd(self, result, reference, voxelspacing = None, connectivity = 1):
        try:
            hd1 = self.surface_distances(result, reference, voxelspacing, connectivity).max()
            hd2 = self.surface_distances(reference, result, voxelspacing, connectivity).max()
            hd = max(hd1, hd2)
        except:
            hd = 0

        return hd

    def hd95(self, result, reference, voxelspacing = None, connectivity = 1):
        hd1 = self.surface_distances(result, reference, voxelspacing, connectivity).max()
        hd2 = self.surface_distances(reference, result, voxelspacing, connectivity).max()
        
        hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
        
        return hd95

    def hausdorff_distance_2d(self): 
        for slc in range(self.length):
            self.GT = self.mask[:, :, slc].sum()
            self.CM = self.prediction[:, :, slc].sum()
            self.TP = (self.mask[:, :, slc] * self.prediction[:, :, slc]).sum()
            self.FN = np.abs(self.GT - self.TP)
            self.FP = np.abs(self.CM - self.TP)
            
            drh = self.hd(self.mask[:, :, slc], self.prediction[:, :, slc], 2, 1)
            # drh = self.hd95(self.mask[:,:,slc], self.prediction[:,:,slc], 2, 1)
            # drh = directed_hausdorff(self.mask[:,:,slc], self.prediction[:,:,slc])
            # drh = max(directed_hausdorff(self.mask[:,:,slc], self.prediction[:,:,slc], 2)[0], directed_hausdorff(self.prediction[:,:,slc], self.mask[:,:,slc], 2)[0])
            print(f'{self.sub_name()} Slice: {self.length - slc} Hausdorff Distance = {drh}')


            # voxel_spacing = np.array(self.mask[:,:,slc].GetSpacing())[::-1]
            # print(voxel_spacing)

    def fpr_2d(self):
        for slc in range(self.length):
            self.GT = self.mask[:, :, slc].sum()
            self.CM = self.prediction[:, :, slc].sum()
            self.TP = (self.mask[:, :, slc] * self.prediction[:, :, slc]).sum()
            self.FN = np.abs(self.GT - self.TP)
            self.FP = np.abs(self.CM - self.TP)        
            self.TN = np.abs(self.CROPP_KERNEL * self.CROPP_KERNEL - self.GT - self.FP)

            print(f'{self.sub_name()} Slice: {self.length - slc} FPR = {self.fpr()}')

        fpr = round(float((self.FP + self.smooth) / (self.FP + self.TN + self.smooth)), 3)

    def dice(self):
        dice = round(float((2 * self.TP + self.smooth) / (2 * self.TP + self.FP + self.FN + self.smooth)), 3)

        return dice

    def recall(self):
        recall = round(float((self.TP + self.smooth) / (self.TP + self.FN + self.smooth)), 3)    

        return recall

    def precision(self):
        precision = round(float((self.TP + self.smooth) / (self.TP + self.FP + self.smooth)), 3)

        return precision

    def fpr(self):
        # self.TN = np.abs(192 * 144 - (self.TP + self.FP + self.FN))
        # fpr = round(float((self.FP + self.smooth) / (self.FP + self.TN + self.smooth)), 3)
        
        tn = int(((self.mask == 0) * (self.prediction == 0)).sum())
        fpr = 1 - (round(float((tn + self.smooth) / (self.FP + tn + self.smooth)), 3))
        
        return fpr

    def hausdorff_distance(self):  
        drh = max(directed_hausdorff(self.mask, self.prediction, 2)[0], directed_hausdorff(self.prediction, self.mask, 2)[0])
        # drh = self.hd(u, v, 2, 2)

        return drh

    def jaccard(self):
        jac = round(float((self.TP + self.smooth) / (self.TP + self.FP + self.FN + self.smooth)), 3)

        return jac

    def tissue_volume(self, matrix):
        fov = CardioCascadeNet.ReadImages(self.path_to_mask).get_nii_fov()
        volume_size = fov[0] * fov[1] * fov[2]
        mask_volume = round(matrix.sum().item() / 1000 * volume_size, 2)

        return mask_volume

    def tissue_volume_2d(self):
        for slc in range(self.length):     
            print(
                f'{self.sub_name()} Slice: {self.length - slc} GT volume = {self.tissue_volume(self.mask[:, :, slc])} ml'
                f'Slice: {self.length - slc} CM volume = {self.tissue_volume(self.prediction[:, :, slc])} ml'
                )

    def pixels_count(self, matrix): 
        return matrix.sum()

    def pixel_count_2d(self):
        for slc in range(self.length):     
            print(
                f'{self.sub_name()} '
                f'Slice: {self.length - slc} '
                f'GT pixels = {self.pixels_count(self.mask[:, :, slc])} '
                f'CM pixels = {self.pixels_count(self.prediction[:, :, slc])}'
                )

    def stat_value(self, name):
        method = getattr(self, name)
        
        return method()

    def stat_value_2d(self, name):
        method = getattr(self, name + '_2d')
        
        return method()

    def __str__(self):
        out_message = ""
        out_message += (f"Statistics was counted for {self.DICT_CLASS[self.num_class]} tissue ")
        out_message += (f'{self.sub_name()}: '\
            f' Mean Dice = {self.dice()}, ' \
            f' Mean Recall = {self.recall()}, '\
            f' Mean Precision = {self.precision()}, '\
            f' Mean Jaccard = {self.jaccard()}, '\
            # f' Mean HD = {self.hausdorff_distance()}, '\
            # f' Mean FPR = {self.fpr()}, '\
            )

        # out_message += (f'{self.sub_name()}: FN pixels {self.FN}, FP pixels {self.FP}, TP pixels: {self.TP}')
        # out_message += (f'{self.sub_name()}: '\
        #     f' GT volume = {self.tissue_volume(self.mask)} ml, '\
        #     f' CM volume = {self.tissue_volume(self.prediction)} ml '\
        #     f' Difference = {round(self.tissue_volume(self.mask) - self.tissue_volume(self.prediction), 2)} ml ')
        
        # out_message += (f'{self.sub_name()}: '\
        #     f' Count of GT pixels = {self.pixels_count(self.mask)} '\
        #     f' Count of CM pixels = {self.pixels_count(self.prediction)} ')

        # out_message += (f'{self.dice_2d()}\n{self.recall_2d()}\n{self.precision_2d()}')
        # out_message += (f'{self.fpr_2d()}\n{self.hausdorff_distance_2d()}')
        # out_message += (f'{self.tissue_volume_2d()}\n{self.pixel_count_2d()}')

        return out_message

        

class CompareBullsEyeMatrix(CardioCascadeNet.MetaParameters):
    def __init__(self, path_to_mask: str, path_to_prediction: str, num_class: int):
        super(CardioCascadeNet.MetaParameters, self).__init__()

        self.path_to_mask = path_to_mask
        self.path_to_prediction = path_to_prediction
        self.num_class = num_class

        self.mask = self.load_matrix(self.path_to_mask, num_class, False, 'etalon')
        self.prediction = self.load_matrix(self.path_to_prediction, num_class, True, 'predict')
        
        self.length = self.mask.shape[-1]

        self.smooth = 1e-5
        self.GT = self.mask.sum()
        self.CM = self.prediction.sum()
        self.TP = (self.mask * self.prediction).sum()
        self.FN = np.abs(self.GT - self.TP)
        self.FP = np.abs(self.CM - self.TP)

    def sub_name(self):
        name = self.path_to_mask.split('/')[-1]
        name = name.rstrip('.nii')
        
        return name

    @staticmethod
    def load_matrix(path_to_matrix:str, key_class:int, cond:bool, auf=None):
        matrix = nib.load(path_to_matrix)
        matrix = np.array(matrix.dataobj)

        # if np.max(matrix) > 17: 
        #     print("ERRRROOOR")

        if np.max(matrix) > 4:
            for slc in range(matrix.shape[2]):
                for basal in range(1, 7):
                    if (matrix[:,:,slc]==basal).any():
                        matrix[:,:,slc][matrix[:,:,slc]<1] = 1
                        matrix[:,:,slc][matrix[:,:,slc]==basal] = 1
                        matrix[:,:,slc][matrix[:,:,slc]!=basal] = 1

                for medial in range(7, 13):
                    if (matrix[:,:,slc]==medial).any():
                        matrix[:,:,slc][matrix[:,:,slc]<1] = 2
                        matrix[:,:,slc][matrix[:,:,slc]==medial] = 2
                        matrix[:,:,slc][matrix[:,:,slc]!=medial] = 2

                for apical in range(13, 17):
                    if (matrix[:,:,slc]==apical).any():
                        matrix[:,:,slc][matrix[:,:,slc]<1] = 3
                        matrix[:,:,slc][matrix[:,:,slc]==apical] = 3
                        matrix[:,:,slc][matrix[:,:,slc]!=apical] = 3

                for apex in range(17, 18):
                    if (matrix[:,:,slc]==apex).any():
                        matrix[:,:,slc][matrix[:,:,slc]<1] = 4
                        matrix[:,:,slc][matrix[:,:,slc]==apex] = 4
                        matrix[:,:,slc][matrix[:,:,slc]!=apex] = 4

        else:
            for slc in range(matrix.shape[2]):
                matrix[:,:,slc][matrix[:,:,slc]!=np.max(matrix[:,:,slc])] = np.max(matrix[:,:,slc])

        matrix[matrix!=key_class] = 0
        matrix[matrix==key_class] = 1

        return matrix            

    def fpr_2d(self):
        for slc in range(self.length):
            self.GT = self.mask[:, :, slc].sum()
            self.CM = self.prediction[:, :, slc].sum()
            self.TP = (self.mask[:, :, slc] * self.prediction[:, :, slc]).sum()
            self.FN = np.abs(self.GT - self.TP)
            self.FP = np.abs(self.CM - self.TP)        
            self.TN = np.abs(self.CROPP_KERNEL * self.CROPP_KERNEL - self.GT - self.FP)

            print(f'{self.sub_name()} Slice: {self.length - slc} FPR = {self.fpr()}')

        fpr = round(float((self.FP + self.smooth) / (self.FP + self.TN + self.smooth)), 3)

    def dice(self):
        dice = round(float((2 * self.TP + self.smooth) / (2 * self.TP + self.FP + self.FN + self.smooth)), 3)

        return dice

    def recall(self):
        recall = round(float((self.TP + self.smooth) / (self.TP + self.FN + self.smooth)), 3)    

        return recall

    def precision(self):
        precision = round(float((self.TP + self.smooth) / (self.TP + self.FP + self.smooth)), 3)

        return precision

    def fpr(self):
        # self.TN = np.abs(192 * 144 - (self.TP + self.FP + self.FN))
        # fpr = round(float((self.FP + self.smooth) / (self.FP + self.TN + self.smooth)), 3)
        
        tn = int(((self.mask == 0) * (self.prediction == 0)).sum())
        fpr = 1 - (round(float((tn + self.smooth) / (self.FP + tn + self.smooth)), 3))
        
        return fpr

    def dice_2d(self):
        list_dsc = []
        for slc in range(self.length):
            self.GT = self.mask[:, :, slc].sum()
            self.CM = self.prediction[:, :, slc].sum()

            # if self.GT == 0:
            #     # print(f'{self.sub_name()} Slice: {self.length - slc} Dice = {0}')
            #     list_dsc.append(0)
            #     pass
            # else:
            self.TP = (self.mask[:, :, slc] * self.prediction[:, :, slc]).sum()
            self.FN = np.abs(self.GT - self.TP)
            self.FP = np.abs(self.CM - self.TP)

            # print(f'{self.sub_name()}: FN pixels {self.FN}, FP pixels {self.FP}, TP pixels: {self.TP}')
            # print(f'{self.sub_name()} Slice: {self.length - slc} Dice = {self.dice()}')
            list_dsc.append(self.dice())

        print(f"Sub {self.sub_name()}: {list_dsc}")

        return list_dsc

    def recall_2d(self):
        list_rcl = []

        for slc in range(self.length):
            self.GT = self.mask[:, :, slc].sum()
            self.CM = self.prediction[:, :, slc].sum()
            self.TP = (self.mask[:, :, slc] * self.prediction[:, :, slc]).sum()
            self.FN = np.abs(self.GT - self.TP)
            self.FP = np.abs(self.CM - self.TP)
            
            # print(f'{self.sub_name()} Slice: {self.length - slc} Recall = {self.recall()}')
            list_rcl.append(self.recall())
        
        return list_rcl

    def precision_2d(self):
        list_prsn = []

        for slc in range(self.length):
            self.GT = self.mask[:, :, slc].sum()
            self.CM = self.prediction[:, :, slc].sum()
            self.TP = (self.mask[:, :, slc] * self.prediction[:, :, slc]).sum()
            self.FN = np.abs(self.GT - self.TP)
            self.FP = np.abs(self.CM - self.TP)
            
            # print(f'{self.sub_name()} Slice: {self.length - slc} Precision = {self.precision()}')
            list_prsn.append(self.precision())
        
        return list_prsn

    def tissue_volume(self, matrix):
        fov = CardioCascadeNet.ReadImages(self.path_to_mask).get_nii_fov()
        volume_size = fov[0] * fov[1] * fov[2]
        mask_volume = round(matrix.sum().item() / 1000 * volume_size, 2)

        return mask_volume

    def tissue_volume_2d(self):
        for slc in range(self.length):     
            print(
                f'{self.sub_name()} Slice: {self.length - slc} GT volume = {self.tissue_volume(self.mask[:, :, slc])} ml'
                f'Slice: {self.length - slc} CM volume = {self.tissue_volume(self.prediction[:, :, slc])} ml'
                )

    def pixels_count(self, matrix): 
        return matrix.sum()

    def pixel_count_2d(self):
        for slc in range(self.length):     
            print(
                f'{self.sub_name()} '
                f'Slice: {self.length - slc} '
                f'GT pixels = {self.pixels_count(self.mask[:, :, slc])} '
                f'CM pixels = {self.pixels_count(self.prediction[:, :, slc])}'
                )

    def stat_value(self, name):
        method = getattr(self, name)
        
        return method()

    def stat_value_2d(self, name):
        method = getattr(self, name + '_2d')
        
        return method()

    def __str__(self):
        out_message = ""
        # out_message += (f"Statistics was counted for {self.DICT_CLASS[self.num_class]} tissue ")
        out_message += (f'{self.sub_name()}: '\
            f' Mean Dice = {self.dice()}, ' \
            f' Mean Recall = {self.recall()}, '\
            f' Mean Precision = {self.precision()}, '\
            f' Mean Jaccard = {self.jaccard()}, '\
        #     # f' Mean HD = {self.hausdorff_distance()}, '\
        #     # f' Mean FPR = {self.fpr()}, '\
            )

        # out_message += (f'{self.sub_name()}: FN pixels {self.FN}, FP pixels {self.FP}, TP pixels: {self.TP}')
        out_message += (f'{self.sub_name()}: '\
            f' GT volume = {self.tissue_volume(self.mask)} ml, '\
            f' CM volume = {self.tissue_volume(self.prediction)} ml '\
            f' Difference = {round(self.tissue_volume(self.mask) - self.tissue_volume(self.prediction), 2)} ml ')
        
        # out_message += (f'{self.sub_name()}: '\
        #     f' Count of GT pixels = {self.pixels_count(self.mask)} '\
        #     f' Count of CM pixels = {self.pixels_count(self.prediction)} ')

        # out_message += (f'{self.dice_2d()}\n{self.recall_2d()}\n{self.precision_2d()}')
        # out_message += (f'{self.fpr_2d()}\n{self.hausdorff_distance_2d()}')
        # out_message += (f'{self.tissue_volume_2d()}\n{self.pixel_count_2d()}')

        return out_message
