 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.5
Date: 08-08-2025
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""


import cv2
import torch

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from torch import nn
from scipy import ndimage
from matplotlib import pylab as plt
from matplotlib.backends.backend_pdf import PdfPages
from skimage.transform import resize, rescale           #pip install scikit-image
from skimage.transform import resize, rescale, downscale_local_mean

import CardioCascadeNet



class PredictListImages(CardioCascadeNet.MetaParameters):
    def __init__(self, file_path, dataset_path, unet_type = None, mask_type = None):
        super(CardioCascadeNet.MetaParameters, self).__init__()
        self.file_path = file_path
        self.dataset_path = dataset_path
        self.def_coord = None
        self.__unet_type = unet_type
        self.__mask_type = mask_type
    
    @property
    def unet_type(self):
        return self.__unet_type

    @property
    def mask_type(self):
        return self.__mask_type

    def nifti_list(self, masks):
        list_images, list_templates = [], []
        images = CardioCascadeNet.ReadImages(f"{self.dataset_path}{self.file_path.split('/')[-1]}").view_matrix
        templates = images.copy()

        manual_img_shape = images.shape

        if self.mask_type == 'infer_bull_level':
            # templates = CardioCascadeNet.ReadImages(f"{self.MASKS_DIR}/{self.file_path.split('/')[-1]}").view_matrix
            templates = CardioCascadeNet.ReadImages(f"{self.NEW_UNET3_MASKS_PATH}{self.file_path.split('/')[-1]}").view_matrix

        if masks is not None:
            ##Case we are necessary in ETALON cropping
            ###########################################################################
            # masks = CardioCascadeNet.ReadImages(f"{self.MASKS_DIR}/{self.file_path.split('/')[-1]}").view_matrix
            ###########################################################################

            images, masks, templates, self.def_coord = \
            CardioCascadeNet.CroppPreprocessData(images, masks, templates, unet_type = self.unet_type).presegmentation_tissues(None)
        else:
            masks = np.zeros((images.shape))

        for slc in range(images.shape[2]):
            image, mask, template = \
            CardioCascadeNet.PreprocessData(images[:, :, slc], masks[:, :, slc], templates[:, :, slc], unet_type = self.unet_type, mask_type = self.mask_type).preprocessing

            image, mask, template = \
            CardioCascadeNet.MaskPreprocessing(image, mask, template, mask_type = self.mask_type).mask_preprocessing

            list_images.append(image)
            list_templates.append(template)

        return list_images, list_templates, manual_img_shape, self.def_coord

    @staticmethod
    def old_dicom(file_path):
        old_dicom = dicom.dcmread(file_path)
        old_dicom = old_dicom.PatientName

        return old_dicom

    def dicom_array(self, masks = None):
        list_images, list_templates = [], []

        num_slc = len(self.file_path)
        file_name = self.file_path[0].split('/')[-1]
        img_shp = CardioCascadeNet.ReadImages(self.file_path[0]).get_dcm().shape

        images = np.zeros((img_shp[0], img_shp[1], num_slc))

        for slc in range(num_slc):
            folder_name = self.old_dicom(self.file_path[slc])
            images[:, :, slc] = CardioCascadeNet.ReadImages(f"{self.file_path[slc]}").get_dcm()[:, :, 0]
        
        templates = images.copy()
        manual_img_shape = images.shape

        if self.mask_type == 'infer_bull_level':
            templates = np.zeros((img_shp[0], img_shp[1], num_slc))

            for slc in range(num_slc):
                folder_name = self.old_dicom(self.file_path[slc])
                templates[:, :, slc] = CardioCascadeNet.ReadImages(f"{self.NEW_UNET3_MASKS_PATH}{folder_name}/{self.file_path[slc].split('/')[-1]}").get_dcm()[:, :, 0]

        if masks is not None:
            images, masks, templates, self.def_coord = \
            CardioCascadeNet.CroppPreprocessData(images, masks, templates, unet_type = self.unet_type).presegmentation_tissues(None, self.cropp_gap)
        
        else:
            masks = np.zeros((images.shape))

        for slc in range(num_slc):
            image, mask, template = \
            CardioCascadeNet.PreprocessData(images[:, :, slc], masks[:, :, slc], templates[:, :, slc], unet_type = self.unet_type, mask_type = self.mask_type).preprocessing
                        
            image, mask, template = \
            CardioCascadeNet.MaskPreprocessing(image, mask, template, mask_type = self.mask_type).mask_preprocessing

            list_images.append(image)
            list_templates.append(template)

        return list_images, list_templates, manual_img_shape, self.def_coord


class PredictionMask(CardioCascadeNet.MetaParameters):
    def __init__(self, model, images, templates, image_shp, def_coord, unet_type):
        super().__init__()

        self.kernel_size = CardioCascadeNet.ChooseKernelSize().kernel_size(unet_type)    
        self.__device = CardioCascadeNet.device
        self.__model = model
        self.__images = images
        self.__image_shp = image_shp
        self.__templates = templates
        self.__def_coord = def_coord
        self.__unet_type = unet_type

    @property
    def model(self):
        return self.__model

    @property
    def device(self):
        return self.__device

    @property
    def images(self):
        return self.__images

    @property
    def image_shp(self):
        return self.__image_shp

    @property
    def templates(self):
        return self.__templates

    @property
    def def_coord(self):
        return self.__def_coord

    @property
    def unet_type(self):
        return self.__unet_type

    def predict(self, image):
        self.model.eval()

        with torch.no_grad():
            image = np.expand_dims(image, 1)
            image = image.transpose(1, 0, 2, 3)
            image = torch.from_numpy(image).to(self.device)

            predict = torch.softmax(self.model(image), dim = 1)
            predict = torch.argmax(predict, dim = 1).cpu()

        return predict, image

    @staticmethod
    def patch_unfolder(matrix):
        matrix = np.expand_dims(matrix, 0)
        matrix = matrix.transpose(0, 3, 1, 2)
        matrix = torch.from_numpy(matrix)

        kc, kh, kw = 1, 64, 64  # kernel size
        dc, dh, dw = 1, 64, 64  # stride

        matrix = matrix.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
        unfold_shape = matrix.size()
        matrix = matrix.contiguous().view(matrix.size(0), -1, kc, kh, kw)

        matrix = matrix[0, :, 0, :, :]

        return matrix, unfold_shape

    @staticmethod
    def patch_folder(matrix, unfold_shape):
        matrix = np.expand_dims(matrix, 0)
        matrix = np.expand_dims(matrix, 0)
        matrix = torch.from_numpy(matrix)
        patches_manual = matrix.view(unfold_shape)

        output_c = unfold_shape[1] * unfold_shape[4]
        output_h = unfold_shape[2] * unfold_shape[5]
        output_w = unfold_shape[3] * unfold_shape[6]

        patches_manual = patches_manual.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        patches_manual = patches_manual.view(1, output_c, output_h, output_w)
        predict = patches_manual[0, 0, :, :]

        return predict

    @property
    def get_predicted_mask(self):
        mask_list, template_list = [], []
        smooth = 1e-6

        for slc in range(0, len(self.images)):
            image = self.images[slc]
            template = self.templates[slc]
            
            ########################################################################
            # image, unfold_shape = self.patch_unfolder(image)
            # template, unfold_shape_02 = self.patch_unfolder(template)

            # image = np.array(image, dtype = np.float32)
            # template = np.array(template, dtype = np.float32)
            # shp = image.shape

            # image = np.array([image, template], dtype = np.float32)[:, :, :, :]
            # image = image.transpose(1, 0, 2, 3)

            # new_images = np.zeros((shp[0], 1, shp[1], shp[2]))
            
            # for i in range(shp[0]):
            #     img = image[i, :, :, :]
            #     new_images[i, :, :, :] = self.predict(img)[0]

            # predict = self.patch_folder(new_images, unfold_shape)

            # predict = np.reshape(predict, (self.kernel_size, self.kernel_size))
            # predict = np.array(predict, dtype = np.float32)
            ########################################################################

            # if not patch_maker
            ########################################################################
            image = np.array([image, template], dtype = np.float32)[:, :, :, 0]
        
            predict, image = self.predict(image)
            predict = np.reshape(predict, (self.kernel_size, self.kernel_size))
            predict = np.array(predict, dtype = np.float32)

            template = np.reshape(template, (self.kernel_size, self.kernel_size))
            template = np.array(template, dtype = np.float32)

            ########################################################################
            predict = self.threshhold_lv_level(predict)
            predict = self.threshhold_scar(predict)
            predict = self.expand_matrix(predict, self.image_shp[0], self.image_shp[1])
            predict = resize(predict, (self.image_shp[0], self.image_shp[1]), anti_aliasing_sigma = False)

            mask_list.append(predict)

        mask_list = self.postprocess_matrix(mask_list)

        return mask_list

    def threshhold_lv_level(self, predict):
        if self.unet_type == 'lv_level':
            try: 
                unique, counts = np.unique(predict, return_counts = True)
                test_dict = dict(zip(unique, counts))
                lv_level = int(list(test_dict.keys())[0])

                if lv_level != 0:
                    predict[predict != 0] = lv_level
                else:
                    lv_level = int(list(test_dict.keys())[1])
                    predict[predict != 0] = lv_level
            except:
                pass

        return predict

    def threshhold_scar(self, predict):
        try:

            if self.unet_type == 'close_cropp' or self.unet_type == 'cropp':
            # if self.DICT_CLASS[2] == 'MYO' and self.DICT_CLASS[3] == 'FIB':
                pred_fib = predict[predict == 3]            
                pred_myo = predict[predict == 2]
                pred_lv = predict[predict == 1]
                
                rel_volume = (pred_fib.sum().item() + 1e-4) / (pred_fib.sum().item() + pred_myo.sum().item() + 1e-4) * 100
                
                if rel_volume < 1 and (predict == 3).sum().item() > 0:
                    predict[predict == 3] = 2

                elif pred_lv.sum().item() / (pred_fib.sum().item() + pred_myo.sum().item() + 1e-4) * 100 < 5:
                    predict[predict == 1] = 3
        except:
            pass

        return predict

    def expand_matrix(self, mask, row_img, column_img):
        new_matrix = np.zeros((row_img, column_img))
        
        ## After prediction of the resized and rescaled image
        if self.def_coord is None:
            row_msk, column_msk = mask.shape
            max_kernel = max(row_img, column_img)
            mask = rescale(mask, (max_kernel / mask.shape[0], max_kernel / mask.shape[1]), anti_aliasing = False, order = 0)
            new_matrix = mask[: row_img, : column_img]

        ## After prediction of cropped and rescaled image
        elif self.def_coord is not None:
            X = (self.def_coord[0] - self.CROPP_KERNEL // 2)
            Y = (self.def_coord[1] - self.CROPP_KERNEL // 2)
            new_matrix[X: X + self.CROPP_KERNEL, Y: Y + self.CROPP_KERNEL] = mask

        return new_matrix

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


class NiftiSaver(CardioCascadeNet.MetaParameters):
    def __init__(self, masks_list, file_path, inference_directory):         
        super(CardioCascadeNet.MetaParameters, self).__init__()

        self.__masks_list = masks_list
        self.__inference_directory = inference_directory
        self.__file_name = file_path.split('/')[-1]

    @property
    def masks_list(self):
        return self.__masks_list

    @property
    def inference_directory(self):
        return self.__inference_directory

    @property
    def file_name(self):
        return self.__file_name

    @property
    def save_nifti(self):
        new_image = nib.Nifti1Image(self.masks_list, affine = np.eye(4))
        nib.save(new_image, f'{self.inference_directory}/{self.file_name}')


class DicomSaver(CardioCascadeNet.MetaParameters):
    def __init__(self, masks_list, file_path, inference_directory):         
        super(CardioCascadeNet.MetaParameters, self).__init__()

        self.masks_list = masks_list
        self.file_name = file_path
        self.inference_directory = inference_directory
        self.manual_dir = self.NEW_DATA_PATH

    def old_dicom(self):
        old_dicom = dicom.dcmread(self.file_name)

        return old_dicom

    def change_name(self, old_dicom):
        seq_name = old_dicom[0x0018, 0x1030]
        seq_name.value += '_Mask'
        seq_number = old_dicom[0x0020, 0x0011]
        seq_number.value = int(seq_number.value) + 1000

        return old_dicom        

    def change_grey_to_color(self, old_dicom):
        old_dicom.PhotometricInterpretation = 'RGB'
        old_dicom.SamplesPerPixel = 3
        old_dicom.BitsAllocated = 8
        old_dicom.BitsStored = 8
        old_dicom.HighBit = 7
        old_dicom.add_new(0x00280006, 'US', 0)

        return old_dicom

    def new_dicom_array(self):
        dcm2 = self.old_dicom().pixel_array

        if len(list(dcm2.shape)) == 2:
            new_dicom_array = cv2.cvtColor(dcm2, cv2.COLOR_GRAY2RGB)
            new_dicom_array = new_dicom_array / 4095 * 255
            new_dicom_array = new_dicom_array.astype(np.uint8)

            # mask = self.masks_list[:, :, 0].astype(np.float16)
            mask = self.masks_list.astype(np.float16)
            
            # new_dicom_array[:, :, 2][mask == 1] += 100
            # new_dicom_array[:, :, 2][mask == 2] -= 150
            # new_dicom_array[:, :, 1][mask == 3] -= 220

            new_dicom_array[:, :, 0][mask == 1] = 51
            new_dicom_array[:, :, 1][mask == 1] = 51
            new_dicom_array[:, :, 2][mask == 1] = 255

            new_dicom_array[:, :, 0][mask == 2] = 204
            new_dicom_array[:, :, 1][mask == 2] = 204
            new_dicom_array[:, :, 2][mask == 2] = 0

            new_dicom_array[:, :, 0][mask == 3] = 0
            new_dicom_array[:, :, 1][mask == 3] = 153
            new_dicom_array[:, :, 2][mask == 3] = 0

            new_dicom_array[:, :, 0][mask == 4] = 151
            new_dicom_array[:, :, 1][mask == 4] = 151
            new_dicom_array[:, :, 2][mask == 4] = 205

            new_dicom_array[:, :, 0][mask == 5] = 204
            new_dicom_array[:, :, 1][mask == 5] = 14
            new_dicom_array[:, :, 2][mask == 5] = 34

            new_dicom_array[:, :, 0][mask == 6] = 244
            new_dicom_array[:, :, 1][mask == 6] = 153
            new_dicom_array[:, :, 2][mask == 6] = 0

            new_dicom_array[:, :, 0][mask == 7] = 91
            new_dicom_array[:, :, 1][mask == 7] = 0
            new_dicom_array[:, :, 2][mask == 7] = 25

            new_dicom_array[:, :, 0][mask == 8] = 47
            new_dicom_array[:, :, 1][mask == 8] = 144
            new_dicom_array[:, :, 2][mask == 8] = 33

            new_dicom_array[:, :, 0][mask == 9] = 33
            new_dicom_array[:, :, 1][mask == 9] = 123
            new_dicom_array[:, :, 2][mask == 9] = 99

            new_dicom_array[:, :, 0][mask == 10] = 26
            new_dicom_array[:, :, 1][mask == 10] = 23
            new_dicom_array[:, :, 2][mask == 10] = 25

            new_dicom_array[:, :, 0][mask == 11] = 24
            new_dicom_array[:, :, 1][mask == 11] = 124
            new_dicom_array[:, :, 2][mask == 11] = 99

            new_dicom_array[:, :, 0][mask == 12] = 0
            new_dicom_array[:, :, 1][mask == 12] = 15
            new_dicom_array[:, :, 2][mask == 12] = 33

            new_dicom_array[:, :, 0][mask == 13] = 204
            new_dicom_array[:, :, 1][mask == 13] = 204
            new_dicom_array[:, :, 2][mask == 13] = 0

            new_dicom_array[:, :, 0][mask == 14] = 0
            new_dicom_array[:, :, 1][mask == 14] = 134
            new_dicom_array[:, :, 2][mask == 14] = 0

            new_dicom_array[:, :, 0][mask == 15] = 2
            new_dicom_array[:, :, 1][mask == 15] = 51
            new_dicom_array[:, :, 2][mask == 15] = 56

            new_dicom_array[:, :, 0][mask == 16] = 77
            new_dicom_array[:, :, 1][mask == 16] = 204
            new_dicom_array[:, :, 2][mask == 16] = 0

            new_dicom_array[:, :, 0][mask == 17] = 0
            new_dicom_array[:, :, 1][mask == 17] = 88
            new_dicom_array[:, :, 2][mask == 17] = 0

        else:
            new_dicom_array = np.zeros((dcm2.shape[0], dcm2.shape[1], 3, dcm2.shape[2]))

            for slc in range(dcm2.shape[2]):
                new_dicom_array[:, :, :, slc] = cv2.cvtColor(dcm2[:, :, slc], cv2.COLOR_GRAY2RGB)

            new_dicom_array = new_dicom_array / 4095 * 255
            new_dicom_array = new_dicom_array.astype(np.uint8)
            mask = self.masks_list[:, :, :].astype(np.float16)
            mask = mask.transpose(2, 1, 0)
            
            mask = np.expand_dims(mask, -2)

            for slc in range(mask.shape[3]):
                masks = mask[:,:,0,slc]
                # new_dicom_array[:, :, 2, slc][masks == 1] = 220
                # new_dicom_array[:, :, 1, slc][masks == 2] = 150
                # new_dicom_array[:, :, 2, slc][masks == 3] = 100

                new_dicom_array[:, :, 0, slc][masks == 1] = 51
                new_dicom_array[:, :, 1, slc][masks == 1] = 51
                new_dicom_array[:, :, 2, slc][masks == 1] = 255

                new_dicom_array[:, :, 0, slc][masks == 2] = 204
                new_dicom_array[:, :, 1, slc][masks == 2] = 204
                new_dicom_array[:, :, 2, slc][masks == 2] = 0
                
                new_dicom_array[:, :, 0, slc][masks == 3] = 0
                new_dicom_array[:, :, 1, slc][masks == 3] = 153
                new_dicom_array[:, :, 2, slc][masks == 3] = 0

            new_dicom_array = new_dicom_array.transpose(0, 1, 3, 2)

        return new_dicom_array

    def new_dicom_array_3d(self):
        dcm2 = self.old_dicom().pixel_array
        # dcm2 = dcm2.transpose(2, 1, 0)

        new_dicom_array = np.zeros((dcm2.shape[0], dcm2.shape[1], 3, dcm2.shape[2]))

        for slc in range(dcm2.shape[2]):
            new_dicom_array[:, :, :, slc] = cv2.cvtColor(dcm2[:, :, slc], cv2.COLOR_GRAY2RGB)

        # new_dicom_array = cv2.cvtColor(dcm2, cv2.COLOR_GRAY2RGB)
        new_dicom_array = new_dicom_array / 4095 * 255
        new_dicom_array = new_dicom_array.astype(np.uint8)
        mask = self.masks_list[:, :, :].astype(np.float16)
        mask = mask.transpose(2, 1, 0)
        
        mask = np.expand_dims(mask, -2)

        for slc in range(mask.shape[3]):
            msk = mask[:, :, 0, slc]
            new_dicom_array[:, :,2, slc][msk == 1] = 220
            new_dicom_array[:, :,1, slc][msk == 2] = 150
            new_dicom_array[:, :,2, slc][msk == 3] = 100

        new_dicom_array = new_dicom_array.transpose(0, 1, 3, 2)

        return new_dicom_array

    def change_value_range_info(self, old_dicom):
        old_dicom.SmallestImagePixelValue = np.min(self.new_dicom_array())
        old_dicom.LargestImagePixelValue = np.max(self.new_dicom_array())

        return old_dicom

    def dicom_file_name(self):
        new_file_name = self.file_name.split('/')[-1]
        
        return new_file_name

    def save_dicom_mask(self):
        old_dicom = self.change_name(self.old_dicom())
        
        # dcm2 = self.old_dicom().pixel_array
        
        if len(list(old_dicom.pixel_array.shape)) == 2:
            # mask = self.masks_list.astype(np.float16)
            mask = self.masks_list.astype(np.int16)
            old_dicom.PixelData = mask.tostring()
        else:
            mask = self.masks_list.astype(np.float16)
            mask = mask.transpose(2, 1, 0)
            old_dicom.PixelData = mask.tostring()

        # mask = self.masks_list[:,:,0].astype(np.float16)
        # old_dicom.PixelData = mask.tostring()
        new_dir_name = old_dicom.PatientName

        fdwr.create_dir(project_name = f"{self.inference_directory}/{new_dir_name}")
        old_dicom.save_as(f"{self.inference_directory}/{new_dir_name}/{self.dicom_file_name()}")

    def save_dicom_mask_3d(self):
        old_dicom = self.change_name(self.old_dicom())
        mask = self.masks_list[:,:,:].astype(np.float16)
        mask = mask.transpose(2, 1, 0)
        old_dicom.PixelData = mask.tostring()
        new_dir_name = old_dicom.PatientName           
        fdwr.create_dir(project_name = f'{self.inference_directory}/{new_dir_name}')
        old_dicom.save_as(f'{self.inference_directory}/{new_dir_name}/{self.dicom_file_name()}')

    def save_dicom(self):
        old_dicom = self.change_name(self.old_dicom())
        old_dicom = self.change_grey_to_color(old_dicom)
        old_dicom = self.change_value_range_info(old_dicom)

        old_dicom.PixelData = self.new_dicom_array().tostring()

        new_dir_name = old_dicom.PatientName

        fdwr.create_dir(project_name = f"{self.inference_directory}/{new_dir_name}")
        old_dicom.save_as(f"{self.inference_directory}/{new_dir_name}/{self.dicom_file_name()}")


class PdfSaver(CardioCascadeNet.MetaParameters):
    def __init__(self, file_path, dataset_path, inference_directory):
        super(CardioCascadeNet.MetaParameters, self).__init__()

        self.dataset_path = dataset_path
        self.inference_directory = inference_directory
        self.file_name = file_path.split('/')[-1]
        
        self.images_list = CardioCascadeNet.ReadImages(f"{self.dataset_path}{self.file_name}").view_matrix
        self.fibmasks_list = CardioCascadeNet.ReadImages(f"{self.NEW_UNET2_MASKS_PATH}{self.file_name}").view_matrix
        
        self.images_list = self.images_list.transpose(2, 0, 1)
        self.fibmasks_list = self.fibmasks_list.transpose(2, 0, 1)
        
        if self.UNET4 and self.UNET5:
            try:
                self.bullmasks_list = CardioCascadeNet.ReadImages(f"{self.inference_directory}/{self.file_name}").view_matrix
                self.manual_bullmasks_list = CardioCascadeNet.ReadImages(f"{self.MASKS_DIR}_bullmasks/{self.file_name}").view_matrix        
                self.manual_fibmasks_list = CardioCascadeNet.ReadImages(f"{self.MASKS_DIR}/{self.file_name}").view_matrix        
            
                self.bullmasks_list = self.bullmasks_list.transpose(2, 0, 1)        
                self.manual_fibmasks_list = self.manual_fibmasks_list.transpose(2, 0, 1)
                self.manual_bullmasks_list = self.manual_bullmasks_list.transpose(2, 0, 1)

            except Exception as e:
                print(f"ERROR {e} while loading bullmasks")

        self.smooth = 1e-5
        self.rows = 3
        self.bbox = dict(boxstyle = "round", fc = "0.8")
        self.arrowprops = dict(arrowstyle = "->", connectionstyle = "angle, angleA = 0, angleB = 90,rad = 10")

    @property
    def create_dict_volume_class(self):
        volume_dict = {}
    
        for key in range(1, self.NUM_CLASS):
            volume_dict[f'Volume_{self.DICT_CLASS[key]}'] = []
            volume_dict[f'Chunk_{self.DICT_CLASS[key]}'] = []

        return volume_dict

    @property
    def get_stats_parameters(self):
        volume_list_dict = {}
        fov = CardioCascadeNet.ReadImages(f"{self.dataset_path}{self.file_name}").get_nii_fov()
        volume_size = fov[0] * fov[1] * fov[2]

        for key in range(1, self.NUM_CLASS):
            volume_list_dict[f'Volume_{self.DICT_CLASS[key]}'] = []

        for mask in self.fibmasks_list:
            for key in range(1, self.NUM_CLASS):
                mask_layer = (mask == key)
                volume_list_dict[f'Volume_{self.DICT_CLASS[key]}'].append(round((mask_layer.sum()) * volume_size, 0))

        return volume_list_dict

    @staticmethod
    def divide_chunks(input_list, rows):
        for i in range(0, len(input_list), rows):
            yield input_list[i:i + rows]

    def preprocess_matrix(self, matrix):
        matrix  = np.flip(matrix, (1))
        matrix = np.rot90(matrix, k = 1, axes = (0, 1))
        
        return matrix

    def add_annotate_class(self, slc, ax, mask_slc):
        if np.max(mask_slc) <= 4:
            for clss in range(1, 4):
                if mask_slc[mask_slc == clss].sum().item() > 3:
                    mark_mask = mask_slc.copy()
                    mark_mask[mark_mask != clss] = 0

                    if clss == 1:
                        weight_mass_y, weight_mass_x = ndimage.measurements.center_of_mass(mark_mask)

                    else:
                        clsrf = CardioCascadeNet.InstancesFinder(mark_mask, kernel = np.min(mark_mask.shape), num_class = clss)
                        clusters = clsrf.find_clusters()
                        max_size = 0
                        
                        for i in range(len(clusters[:])):
                            cluster_size = len(clusters[i]['coords'])
                            
                            if cluster_size > len(clusters[max_size]['coords']):
                                max_size = i

                        weight_mass_y, weight_mass_x = clusters[max_size]['coords'][len(clusters[max_size]['coords'])//2]

                    ax.annotate(f'{clss}', 
                                xy = (weight_mass_x, weight_mass_y), 
                                fontsize = 6, xytext = (weight_mass_x + 5, weight_mass_y + 5), 
                                arrowprops = self.arrowprops,
                                bbox = self.bbox, 
                                color = 'black')

                    ax.plot([weight_mass_x], [weight_mass_y],  marker = ".", color = 'orange')

            for key in range(1, 4): 
                mask_slc[0][key - 1] = key

            pass

        else:
            for clss in range(self.NUM_CLASS):
                if mask_slc[mask_slc == clss].sum().item() > 3:
                    mark_mask = mask_slc.copy()
                    mark_mask[mark_mask != clss] = 0
                    weight_mass_y, weight_mass_x = ndimage.measurements.center_of_mass(mark_mask)

                    ax.annotate(f'S{clss}', 
                                xy = (weight_mass_x, weight_mass_y), 
                                fontsize = 6, xytext = (weight_mass_x + 5, weight_mass_y - 5), 
                                #  arrowprops = dict(facecolor = 'red'),
                                arrowprops = self.arrowprops,
                                bbox = self.bbox, 
                                color = 'black')

                    ax.plot([weight_mass_x], [weight_mass_y],  marker = ".", color = 'orange')
                    
            for key in range(1, self.NUM_CLASS): 
                mask_slc[0][key - 1] = key

        return ax, mask_slc

    def threshold_scar(self, report_title, page = None, slc = None, class_volume = None):
        try:
            if self.DICT_CLASS[2] == 'MYO' and self.DICT_CLASS[3] == 'FIB':
                if page != None and slc != None:
                    MYOv = class_volume[f'Chunk_{self.DICT_CLASS[2]}'][page]
                    FIBv = class_volume[f'Chunk_{self.DICT_CLASS[3]}'][page]
                    relVolume = round((FIBv[slc] / (FIBv[slc] + MYOv[slc] + self.smooth)) * 100, 1)
                    report_title += (f'RelVol of FIB: {relVolume} % ')

                elif page == None and slc == None:
                    related_full_fib_volume = round((
                        (sum(class_volume[f'Volume_{self.DICT_CLASS[3]}'])) / 
                        (sum(class_volume[f'Volume_{self.DICT_CLASS[2]}']) + 
                            sum(class_volume[f'Volume_{self.DICT_CLASS[3]}']) + self.smooth)) * 100, 0)
                    report_title += f'Full relative volume: ≈ {related_full_fib_volume} %'

        except:
            pass

        return report_title

    def write_class_volume(self, report_title, page = None, slc = None, class_volume = None):
        for key in range(1, self.NUM_CLASS):
            
            if page != None and slc != None:
                chunk_volume = round(class_volume[f"Chunk_{self.DICT_CLASS[key]}"][page][slc] / 1000, 2)

                if chunk_volume > 1:
                    report_title += (
                        f'{self.DICT_CLASS[key]}: {chunk_volume} ml, ' )
                
            elif page == None and slc == None:
                report_title += (
                    f'Full {self.DICT_CLASS[key]} volume: {sum(class_volume[f"Volume_{self.DICT_CLASS[key]}"]) / 1000} ml, \n' )

        return report_title

    def change_17seg_classes(self, bull_mask):
        bull_mask[bull_mask==7] = 1
        bull_mask[bull_mask==8] = 2
        bull_mask[bull_mask==9] = 3
        bull_mask[bull_mask==10] = 4
        bull_mask[bull_mask==11] = 5
        bull_mask[bull_mask==12] = 6

        bull_mask[bull_mask==13] = 1
        bull_mask[bull_mask==14] = 2
        bull_mask[bull_mask==15] = 4
        bull_mask[bull_mask==16] = 5

        bull_mask[bull_mask==17] = 2

        return bull_mask

    @property
    def save_pdf(self):
        volume_list_dict = self.get_stats_parameters
        volume_dict_class = self.create_dict_volume_class
        
        for key in range(1, self.NUM_CLASS):
            volume_dict_class[f'Volume_{self.DICT_CLASS[key]}'] = volume_list_dict[f'Volume_{self.DICT_CLASS[key]}']

        num_chunk = len(self.images_list) % self.rows
        chunk_list_images = list(self.divide_chunks(self.images_list, self.rows))
        chunk_list_fibmasks = list(self.divide_chunks(self.fibmasks_list, self.rows))

        if self.UNET4 and self.UNET5: 
            chunk_list_bullmasks = list(self.divide_chunks(self.bullmasks_list, self.rows))
            chunk_list_manual_fibmasks = list(self.divide_chunks(self.manual_fibmasks_list, self.rows))
            chunk_list_manual_bullmasks = list(self.divide_chunks(self.manual_bullmasks_list, self.rows))

        for key in range(1, self.NUM_CLASS): 
            volume_dict_class[f'Chunk_{self.DICT_CLASS[key]}'] = list(self.divide_chunks(volume_dict_class[f'Volume_{self.DICT_CLASS[key]}'], self.rows))

        num_pages = len(chunk_list_images)
        pp = PdfPages(f'{self.inference_directory}/{self.file_name}_results.pdf')
        
        for page in range(num_pages):
            images = chunk_list_images[page]
            fibmasks = chunk_list_fibmasks[page]
            
            if self.UNET4 and self.UNET5:
                try:
                    bullmasks = chunk_list_bullmasks[page]
                    manual_fibmasks = chunk_list_manual_fibmasks[page]
                    manual_bullmasks = chunk_list_manual_bullmasks[page]
                except Exception as e:
                    print(f"ERROR {e} while loading bullmasks")

            images_on_page = len(fibmasks)
            
            if images_on_page > 1:
                num_images = images_on_page
            elif images_on_page == 1:
                num_images = 3

            if self.UNET4 and self.UNET5: 
                figure, ax = plt.subplots(nrows = num_images, ncols = 4, figsize = (10, 10))
            else:
                figure, ax = plt.subplots(nrows = num_images, ncols = 2, figsize = (10, 10))

            colormap = plt.get_cmap('viridis')  # 'plasma' or 'viridis'
            colormap.set_under('k', alpha = .5)

            for slc in range(images_on_page):                    
                image_slc = self.preprocess_matrix(images[slc])
                fibmask_slc  = self.preprocess_matrix(fibmasks[slc])

                if self.UNET4 and self.UNET5: 
                    try:
                        bullmask_slc = self.preprocess_matrix(bullmasks[slc])
                        manual_fibmask_slc = self.preprocess_matrix(manual_fibmasks[slc])
                        manual_bullmask_slc = self.preprocess_matrix(manual_bullmasks[slc])
                    except Exception as e:
                        print(f"ERROR {e} while loading bullmasks")

                ax[slc, 1], fibmask_slc = self.add_annotate_class(slc, ax[slc, 1], fibmask_slc)
                
                if self.UNET4 and self.UNET5:
                    try:
                        ax[slc, 0], manual_fibmask_slc = self.add_annotate_class(slc, ax[slc, 0], manual_fibmask_slc)
                        ax[slc, 2], bullmask_slc = self.add_annotate_class(slc, ax[slc, 2], bullmask_slc)
                        ax[slc, 3], manual_bullmask_slc = self.add_annotate_class(slc, ax[slc, 3], manual_bullmask_slc)

                        bullmask_slc = self.change_17seg_classes(bullmask_slc)
                        manual_bullmask_slc = self.change_17seg_classes(manual_bullmask_slc)
                    except Exception as e:
                        print(f"ERROR {e} while loading bullmask and manual_fibmask")

                ax[slc, 0].imshow(image_slc, plt.get_cmap('gray'))

                try:
                    ax[slc, 0].imshow(manual_fibmask_slc, alpha = 0.2, interpolation = None, cmap = colormap,  vmin = 0.5)
                    ax[slc, 0].contour(manual_fibmask_slc, alpha = 0.9, cmap = colormap,  vmin = 0.5)
                    ax[slc, 0].set_title('manual_fib_mask')
                except Exception as e:
                    print(f"ERROR {e} while loading manual_fibmask_slc")

                ax[slc, 1].imshow(image_slc, plt.get_cmap('gray'))
                ax[slc, 1].imshow(fibmask_slc, alpha = 0.5, interpolation = None, cmap = colormap,  vmin = 0.5)
                ax[slc, 1].contour(fibmask_slc, alpha = 0.5)
                ax[slc, 1].set_title('fib_mask')


                if self.UNET4 and self.UNET5: 
                    try:
                        ax[slc, 2].imshow(image_slc, plt.get_cmap('gray'))
                        ax[slc, 2].imshow(bullmask_slc, alpha = 0.2, interpolation = None, cmap = colormap,  vmin = 0.5)
                        ax[slc, 2].contour(bullmask_slc, alpha = 0.9, cmap = colormap,  vmin = 0.5)
                        ax[slc, 2].set_title('bull_mask')

                        ax[slc, 3].imshow(image_slc, plt.get_cmap('gray'))
                        ax[slc, 3].imshow(manual_bullmask_slc, alpha = 0.2, interpolation = None, cmap = colormap,  vmin = 0.5)
                        ax[slc, 3].contour(manual_bullmask_slc, alpha = 0.9, cmap = colormap,  vmin = 0.5)
                        ax[slc, 3].set_title('manual_bull_mask')
                    except Exception as e:
                        print(f"ERROR {e} while loading bull_mask and manual_bull_mask")

                report_title = ''
                report_title = self.threshold_scar(report_title, page, slc, volume_dict_class)
                # report_title = self.write_class_volume(report_title, page, slc, volume_dict_class)

                ax[slc, 1].set_title(report_title, fontsize = 8, fontweight = 'bold', loc = 'right')

                figure.tight_layout()
            pp.savefig(figure)

        report_title = ''
        report_title = self.threshold_scar(report_title, None, None, volume_dict_class)
        # report_title = self.write_class_volume(report_title, None, None, volume_dict_class)

        fig = plt.figure(figsize = (8, 8))
        text = fig.text(0.2, 0.7, report_title, ha = 'left', va = 'top', size = 14)

        text.set_path_effects([path_effects.Normal()])
        pp.savefig(fig)
        
        pp.close()
