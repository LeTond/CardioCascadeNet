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
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset
from multiprocessing import Pool, TimeoutError, current_process


import CardioCascadeNet



class GetData(CardioCascadeNet.MetaParameters):
    def __init__(self, files = None, augmentation = None):
        super(CardioCascadeNet.MetaParameters, self).__init__()
        self.files = files
        self.augmentation = augmentation
        
    @property
    def unet_type(self):
        if self.UNET5 is True:
            return 'cropp'
        elif self.UNET4 is True and self.UNET5 is False:
            return 'cropp'
        elif self.UNET3 is True and self.UNET4 is False:
            return 'close_cropp'
        elif self.UNET2 is True and self.UNET3 is False:
            return 'cropp'
        elif self.UNET1 is True and self.UNET2 is False:
            return 'default'
        else:
            raise ValueError 

    @property
    def mask_type(self):
        if self.BGCROPP is True:
            return 'bgcrop'
        elif self.LVCROPP is True:
            return 'lvcropp'
        elif self.BGLVCROPP is True:
            return 'bglvcropp'
        elif self.UNET4 is True and self.UNET5 is False:
            return 'myo_level'
        elif self.UNET5 is True:
            return 'train_bull_level'
        else:
            return None

    @property
    def create_dict_class(self):
        dict_class_stats = {}
  
        dict_class_stats.update(
                {
                    f'{self.DICT_CLASS[key]}' : 
                        {'Subjects': 0, 'pixels': 0} for key in range(1, self.NUM_CLASS)
                }
            )

        return dict_class_stats

    def count_pathology(self, sub_names):
        diction = self.create_dict_class

        for sub_name in sub_names:
            if sub_name.endswith('.nii'):
                masks = CardioCascadeNet.ReadImages(f"{self.MASKS_DIR}/{sub_name}").view_matrix

                for key in range(1, self.NUM_CLASS):
                    if (masks == key).any():
                        diction[f'{self.DICT_CLASS[key]}'].update(
                            {
                                'Subjects': diction[f'{self.DICT_CLASS[key]}']['Subjects'] + 1,
                                'pixels': diction[f'{self.DICT_CLASS[key]}']['pixels'] + masks[masks == key].sum().item()
                            }
                        )

        return diction

    @property
    def cropp_gap(self):
        if self.MULTYGAP:
            cropp_gap = random.choice([6, 7, 8, 9, 10])
        else:
            cropp_gap = 8

        return cropp_gap

    def check_mask(self, mask, sub_name, slc):
        if self.EMPTY is False and mask[mask > 0].sum().item() == 0:
            print(f"Subject {sub_name} slice {slc} was passed because EMPY is FALSE")
            return False
        elif (mask > (self.NUM_CLASS - 1)).any():
            print(f"Subject {sub_name} slice {slc} has class out of range class {self.NUM_CLASS}")
            return False
        else:
            return True

    @staticmethod
    def patch_unfolder(matrix):
        matrix = np.expand_dims(matrix, 0)
        matrix = matrix.transpose(0, 3, 1, 2)
        matrix = torch.from_numpy(matrix)

        kc, kh, kw = 1, 64, 64  # kernel size
        dc, dh, dw = 1, 64, 64  # stride

        matrix = matrix.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
        matrix = matrix.contiguous().view(matrix.size(0), -1, kc, kh, kw)

        matrix = matrix[0, :, 0, :, :]

        return matrix

    def pool_worker(self, file_name):
        list_images, list_masks, list_templates, list_names = [], [], [], []

        if file_name.endswith('.nii'):
            images = CardioCascadeNet.ReadImages(f"{self.ORIGS_DIR}/{file_name}").view_matrix
            masks = CardioCascadeNet.ReadImages(f"{self.MASKS_DIR}/{file_name}").view_matrix

            sub_name = file_name.replace('.nii', '')

            if self.mask_type == 'train_bull_level':
                templates = masks.copy()
            else:
                templates = images.copy()

            if self.unet_type == 'cropp' or self.unet_type == 'close_cropp':
                try:
                    images, masks, templates, def_coord = \
                    CardioCascadeNet.CroppPreprocessData(images, masks, templates, unet_type = self.unet_type).presegmentation_tissues(None, self.cropp_gap)
                except:
                    print(f'Data INFER Preprocessing Problem with {sub_name}')

            for slc in range(images.shape[2]):
                image = images[:, :, slc]
                mask = masks[:, :, slc]
                template = templates[:, :, slc]

                try:
                    image, mask, template = \
                    CardioCascadeNet.PreprocessData(image, mask, template, unet_type = None, mask_type = self.mask_type).preprocessing
                except:
                    print(f'Data Preprocessing Problem with {sub_name}')
                
                ##TODO: bug?? with rotation template while UNET5 training???
                # try:
                #     image, mask, template = \
                #     Augmentation(image, mask, template, unet_type = self.unet_type).rotate_2d     #bug?? with rotation template while UNET5 training???
                #     # image, mask, template = \
                #     # Augmentation(image, mask, template, unet_type = self.unet_type).gauss_noise
                #     # image, mask, template = \
                #     # Augmentation(image, mask, template, unet_type = self.unet_type).rician_noise
                # except:
                #     print(f'Data Augmentation Problem with {sub_name}')

                try:
                    image, mask, template = \
                    CardioCascadeNet.MaskPreprocessing(image, mask, template, mask_type = self.mask_type).mask_preprocessing
                except:
                    print(f'Data MaskPreprocessing Problem with {sub_name}')

                if self.check_mask(mask, sub_name, slc):    
                    # image = self.patch_unfolder(image)
                    # template = self.patch_unfolder(template)
                    # mask = self.patch_unfolder(mask)

                    # for i in range(9):
                    #     list_images.append(image[i, :, :])
                    #     list_masks.append(mask[i, :, :])
                    #     list_templates.append(template[i, :, :])
                    #     list_names.append(f'{sub_name} Slice {images.shape[2] - slc}')
                    ########################################################################
                    list_images.append(image)
                    list_masks.append(mask)
                    list_templates.append(template)
                    list_names.append(f'{sub_name} Slice {images.shape[2] - slc}')
                    ########################################################################

        return list_images, list_masks, list_templates, list_names

    @property
    def generated_data_list(self):
        list_images, list_masks, list_templates, list_names = [], [], [], []

        # print(self.count_pathology(self.files))

        # for subject in self.files:
        #     try:
        #         images, masks, templates, sub_names = self.pool_worker(subject)
        #         for slc in range(len(images)): 
        #             list_images.append(images[slc])
        #             list_masks.append(masks[slc])
        #             list_templates.append(templates[slc])
        #             list_names.append(sub_names[slc])
        #     except:
        #         pass 

        # if self.AUGMENTATION and self.augmentation:
        #     for subject in self.files:
        #         try:
        #             images, masks, templates, sub_names = self.pool_worker(subject)
        #             for slc in range(len(images)): 
        #                 list_images.append(images[slc])
        #                 list_masks.append(masks[slc])
        #                 list_templates.append(templates[slc])
        #                 list_names.append(sub_names[slc])
        #         except:
        #             pass 

        for case in range(1):
            with Pool(processes=4) as pool:
                try:
                    for patch in pool.imap_unordered(self.pool_worker, self.files):
                        size_img = len(patch[0])

                        for slc in range(size_img):
                            list_images.append(patch[0][slc])
                            list_masks.append(patch[1][slc])
                            list_templates.append(patch[2][slc])
                            list_names.append(patch[3][slc])
            
                except:
                    pass 

        # for i in range(1):
        #     with Pool(processes=4) as pool:
        #         try:
        #             for patch in pool.imap_unordered(self.pool_worker, self.files):
        #                 # if self.check_mask(patch[1]):
        #                 list_images.append([img for img in patch[0]])
        #                 list_masks.append([msk for msk in patch[1]])
        #                 list_names.append([nm for nm in patch[2]])
                            
        #         except:
        #             pass 

        try:
            list_images, list_masks, list_templates, list_names = \
            CardioCascadeNet.PreprocessData(list_images, list_masks, list_templates, list_names).shuff_dataset
        except:
            print('Shuffle was broken')
            pass

        return list_images, list_masks, list_templates, list_names


class MyDataset(Dataset, CardioCascadeNet.MetaParameters):
    def __init__(self, ds_images, ds_masks, ds_templates, ds_names, transform = None, images_and_labels = []):
        super().__init__()

        self.kernel_size = CardioCascadeNet.ChooseKernelSize().kernel_size(unet_type = None)
        self.transform = transform
        self.images_and_labels = images_and_labels
        self.images = ds_images
        self.masks = ds_masks
        self.templates = ds_templates
        self.names = ds_names

        for i in range(len(self.images)):
            self.images_and_labels.append((i, i, i, i))

    def preprocessing(self, image, mask, template):
        image = TF.to_pil_image(image)
        image = TF.pil_to_tensor(image)

        mask = mask / self.NUM_CLASS
        mask = TF.to_pil_image(mask)
        mask = TF.pil_to_tensor(mask)

        template = TF.to_pil_image(template)
        template = TF.pil_to_tensor(template)

        tcat = torch.cat((image, mask, template), 0)
        image, mask, template = self.transform(tcat)

        image = np.array(image.reshape(self.kernel_size, self.kernel_size, 1), dtype = np.float32)
        mask = np.array(mask.reshape(self.kernel_size, self.kernel_size, 1), dtype = np.float32)
        template = np.array(template.reshape(self.kernel_size, self.kernel_size, 1), dtype = np.float32)

        mask = np.round(mask * self.NUM_CLASS)
        
        return image, mask, template
        
    def __getitem__(self, item):
        imgs, labs, templs, sub_nms = self.images_and_labels[item]
        image = self.images[imgs][:][:]
        mask = self.masks[labs][:][:]
        sub_names = self.names[sub_nms]
        
        template = self.templates[templs][:][:]
        image, mask, template = self.preprocessing(image, mask, template)

        image = np.array([image, template], dtype = np.float32)[:, :, :, 0]

        mask = np.resize(mask, (self.kernel_size, self.kernel_size))
        mask = np.array(mask, dtype = np.int8)
        mask = np.eye(self.NUM_CLASS)[mask]
        mask = np.array(mask, dtype = np.float32)
        mask = mask.transpose(2, 0, 1)

        return image, mask, sub_names

    def __len__(self):
        return len(self.images)

