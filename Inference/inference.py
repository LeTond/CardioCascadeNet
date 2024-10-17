 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.2
Date: 03-09-2024
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""


from torch import nn

import torch
import cv2
import numpy as np
import nibabel as nib

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from matplotlib import pylab as plt
from matplotlib.backends.backend_pdf import PdfPages
from skimage.transform import resize, rescale       #pip install scikit-image
from skimage.transform import resize, rescale, downscale_local_mean

from Model.unet2D import UNet_2D, UNet_2D_AttantionLayer
from Preprocessing.preprocessing import *
from Postprocessing.postprocessing import *
from configuration import *


class GetListImages(MetaParameters):
    def __init__(self, file_path, path_to_data, dataset_path, unet_type = None, mask_type = None):
        super(MetaParameters, self).__init__()
        self.file_path = file_path
        self.file_name = file_path.split('/')[-1]
        self.path_to_data = path_to_data
        self.dataset_path = dataset_path
        self.def_coord = None
        self.__unet_type = unet_type
        self.__mask_type = mask_type
        self.cropp_gap = 8
    
    @property
    def unet_type(self):
        return self.__unet_type

    @property
    def mask_type(self):
        return self.__mask_type

    def nifti_list(self, masks):
        list_images, list_templates = [], []
        images = ReadImages(f"{self.dataset_path}{self.file_name}").view_matrix
        templates = images.copy()

        orig_img_shape = images.shape

        if self.mask_type == 'infer_bull_level':
            templates = ReadImages(f'./Dataset/HCM_adult_mask/{self.file_name}').view_matrix
            # templates = ReadImages(f'./Dataset/ALMAZ_Unet3_mask_new/{self.file_name}').view_matrix

        if masks is not None:
            images, masks, templates, self.def_coord = \
            CroppPreprocessData(images, masks, templates, unet_type = self.unet_type).presegmentation_tissues(None, self.cropp_gap)
        else:
            masks = np.zeros((images.shape))

        for slc in range(images.shape[2]):
            image, mask, template = \
            PreprocessData(images[:, :, slc], masks[:, :, slc], templates[:, :, slc], unet_type = self.unet_type, mask_type = self.mask_type).preprocessing

            image, mask, template = \
            MaskPreprocessing(image, mask, template, mask_type = self.mask_type).mask_preprocessing

            list_images.append(image)
            list_templates.append(template)

        return list_images, list_templates, orig_img_shape, self.def_coord

    @staticmethod
    def old_dicom(file_path):
        old_dicom = dicom.dcmread(file_path)
        old_dicom = old_dicom.PatientName

        return old_dicom

    def dicom_array(self, def_coord = None, masks = None):
        list_images, list_templates = [], []
        folder_name = self.old_dicom(self.file_path)

        images = ReadImages(f"{self.file_path}").get_dcm()
        templates = images.copy()

        orig_img_shape = images.shape

        if self.mask_type == 'infer_bull_level':
            # templates = ReadImages(f'./Dataset/ALMAZ_Unet3_mask_new/{self.file_name}').view_matrix
            templates = ReadImages(f'./Dataset/HCM_adult_Unet2_mask_new/{self.file_name}').view_matrix

        if masks is not None:
            images, masks, templates, def_coord = \
            CroppPreprocessData(images, masks, templates, unet_type = self.unet_type).presegmentation_tissues(def_coord, self.cropp_gap)
        
        else:
            masks = np.zeros((images.shape))

        for slc in range(images.shape[2]):
            image, mask, template = \
            PreprocessData(images[:, :, slc], masks[:, :, slc], templates[:, :, slc], unet_type = self.unet_type, mask_type = self.mask_type).preprocessing
            
            # image, mask, template = \
            # PreprocessData(images[:, :, slc], None, templates[:, :, slc], unet_type = self.unet_type, mask_type = self.mask_type).preprocessing
            
            image, mask, template = \
            MaskPreprocessing(image, mask, template, mask_type = self.mask_type).mask_preprocessing

            list_images.append(image)
            list_templates.append(template)

        return list_images, list_templates, orig_img_shape, def_coord


class PredictionMask(MetaParameters):
    def __init__(self, model, images, templates, image_shp, def_coord, unet_type):
        super().__init__()

        self.__model = model
        self.__device = device
        self.__images = images
        self.__image_shp = image_shp
        self.__templates = templates
        self.__def_coord = def_coord
        self.__unet_type = unet_type
        self.kernel_size = chklsz.kernel_size(unet_type)    

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

    def predict(self, image):
        self.model.eval()

        with torch.no_grad():
            image = np.expand_dims(image, 1)
            image = image.transpose(1, 0, 2, 3)
            image = torch.from_numpy(image).to(self.device)

            predict = torch.softmax(self.model(image), dim = 1)
            predict = torch.argmax(predict, dim = 1).cpu()

        return predict, image

    @property
    def get_predicted_mask(self):
        mask_list = []
        smooth = 1e-6

        for slc in range(0, len(self.images)):
            image = self.images[slc]
            template = self.templates[slc]
            
            image = np.array([image, template], dtype = np.float32)[:, :, :, 0]            
            predict, image = self.predict(image)
            predict = np.reshape(predict, (self.kernel_size, self.kernel_size))
            predict = np.array(predict, dtype = np.float32)
            
            predict = self.threshhold_myo_level(predict)
            # predict = self.threshhold_scar(predict)
            predict = self.expand_matrix(predict, self.image_shp[0], self.image_shp[1])
            predict = resize(predict, (self.image_shp[0], self.image_shp[1]), anti_aliasing_sigma = False)
            
            mask_list.append(predict)

        mask_list = self.postprocess_matrix(mask_list)

        return mask_list

    def threshhold_myo_level(self, predict):
        if self.UNET4 is True and self.UNET5 is False:
            try: 
                unique, counts = np.unique(predict, return_counts = True)
                test_dict = dict(zip(unique, counts))
                myo_level = int(list(test_dict.keys())[0])

                if myo_level != 0:
                    predict[predict != 0] = myo_level
                else:
                    myo_level = int(list(test_dict.keys())[1])
                    predict[predict != 0] = myo_level
            except:
                pass

        return predict

    def threshhold_scar(self, predict):
        try:
            if self.DICT_CLASS[2] == 'MYO' and self.DICT_CLASS[3] == 'FIB':
                pred_fib = predict[predict == 3]            
                pred_myo = predict[predict == 2]
                rel_volume = (pred_fib.sum().item() + 1e-4) / (pred_fib.sum().item() + pred_myo.sum().item() + 1e-4) * 100
                
                if rel_volume < 1 and (predict == 3).sum().item() > 0:
                    predict[predict == 3] = 2
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


class NiftiSaver(MetaParameters):
    def __init__(self, masks_list, file_path, inference_directory):         
        super(MetaParameters, self).__init__()

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


class DicomSaver(MetaParameters):
    def __init__(self, masks_list, file_path, inference_directory):         
        super(MetaParameters, self).__init__()

        self.masks_list = masks_list
        self.file_name = file_path
        self.inference_directory = inference_directory
        self.orig_dir = self.NEW_DATA_PATH

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

            mask = self.masks_list[:,:,0].astype(np.float16)
            
            # new_dicom_array[:,:,2][mask == 1] += 100
            # new_dicom_array[:,:,2][mask == 2] -= 150
            # new_dicom_array[:,:,1][mask == 3] -= 220

            new_dicom_array[:,:,0][mask == 1] = 51
            new_dicom_array[:,:,1][mask == 1] = 51
            new_dicom_array[:,:,2][mask == 1] = 255

            new_dicom_array[:,:,0][mask == 2] = 204
            new_dicom_array[:,:,1][mask == 2] = 204
            new_dicom_array[:,:,2][mask == 2] = 0

            new_dicom_array[:,:,0][mask == 3] = 0
            new_dicom_array[:,:,1][mask == 3] = 153
            new_dicom_array[:,:,2][mask == 3] = 0

        else:
            new_dicom_array = np.zeros((dcm2.shape[0], dcm2.shape[1], 3, dcm2.shape[2]))

            for slc in range(dcm2.shape[2]):
                new_dicom_array[:,:,:,slc] = cv2.cvtColor(dcm2[:,:,slc], cv2.COLOR_GRAY2RGB)

            new_dicom_array = new_dicom_array / 4095 * 255
            new_dicom_array = new_dicom_array.astype(np.uint8)
            mask = self.masks_list[:,:,:].astype(np.float16)
            mask = mask.transpose(2, 1, 0)
            
            mask = np.expand_dims(mask, -2)

            for slc in range(mask.shape[3]):
                masks = mask[:,:,0,slc]
                # new_dicom_array[:,:,2,slc][masks == 1] = 220
                # new_dicom_array[:,:,1,slc][masks == 2] = 150
                # new_dicom_array[:,:,2,slc][masks == 3] = 100

                new_dicom_array[:,:,0,slc][masks == 1] = 51
                new_dicom_array[:,:,1,slc][masks == 1] = 51
                new_dicom_array[:,:,2,slc][masks == 1] = 255

                new_dicom_array[:,:,0,slc][masks == 2] = 204
                new_dicom_array[:,:,1,slc][masks == 2] = 204
                new_dicom_array[:,:,2,slc][masks == 2] = 0
                
                new_dicom_array[:,:,0,slc][masks == 3] = 0
                new_dicom_array[:,:,1,slc][masks == 3] = 153
                new_dicom_array[:,:,2,slc][masks == 3] = 0

            new_dicom_array = new_dicom_array.transpose(0, 1, 3, 2)

        return new_dicom_array

    def new_dicom_array_3d(self):
        dcm2 = self.old_dicom().pixel_array
        # dcm2 = dcm2.transpose(2,1,0)

        new_dicom_array = np.zeros((dcm2.shape[0], dcm2.shape[1], 3, dcm2.shape[2]))

        for slc in range(dcm2.shape[2]):
            new_dicom_array[:,:,:,slc] = cv2.cvtColor(dcm2[:,:,slc], cv2.COLOR_GRAY2RGB)

        # new_dicom_array = cv2.cvtColor(dcm2, cv2.COLOR_GRAY2RGB)
        new_dicom_array = new_dicom_array / 4095 * 255
        new_dicom_array = new_dicom_array.astype(np.uint8)
        mask = self.masks_list[:,:,:].astype(np.float16)
        mask = mask.transpose(2, 1, 0)
        
        mask = np.expand_dims(mask, -2)

        for slc in range(mask.shape[3]):
            msk = mask[:,:,0,slc]
            new_dicom_array[:,:,2,slc][msk == 1] = 220
            new_dicom_array[:,:,1,slc][msk == 2] = 150
            new_dicom_array[:,:,2,slc][msk == 3] = 100

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
            mask = self.masks_list[:,:,0].astype(np.float16)
            old_dicom.PixelData = mask.tostring()
        else:
            mask = self.masks_list[:,:,:].astype(np.float16)
            mask = mask.transpose(2, 1, 0)
            old_dicom.PixelData = mask.tostring()

        # mask = self.masks_list[:,:,0].astype(np.float16)
        # old_dicom.PixelData = mask.tostring()
        new_dir_name = old_dicom.PatientName           
        fdwr.create_dir(project_name = f'{self.inference_directory}/{new_dir_name}')
        old_dicom.save_as(f'{self.inference_directory}/{new_dir_name}/{self.dicom_file_name()}')


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
        fdwr.create_dir(project_name = f'{self.inference_directory}/{new_dir_name}')

        old_dicom.save_as(f'{self.inference_directory}/{new_dir_name}/{self.dicom_file_name()}')


class PdfSaver(MetaParameters):
    def __init__(self, file_path, dataset_path, inference_directory):
        super(MetaParameters, self).__init__()

        self.dataset_path = dataset_path
        self.inference_directory = inference_directory
        self.file_name = file_path.split('/')[-1]
        
        self.images_list = ReadImages(f"{self.dataset_path}{self.file_name}").view_matrix
        self.masks_list = ReadImages(f"{self.inference_directory}/{self.file_name}").view_matrix
        self.orig_masks_list = ReadImages(f"{self.inference_directory}/{self.file_name}").view_matrix
        self.orig_masks_list = ReadImages(f"{self.MASKS_DIR}/{self.file_name}").view_matrix
        # self.fib_masks_list = ReadImages(f"./Dataset/HCM_adult_Unet2_mask_new/{self.file_name}").view_matrix
        
        self.images_list = self.images_list.transpose(2, 0, 1)
        self.masks_list = self.masks_list.transpose(2, 0, 1)
        self.orig_masks_list = self.orig_masks_list.transpose(2, 0, 1)
        # self.fib_masks_list = self.fib_masks_list.transpose(2, 0, 1)

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
        fov = ReadImages(f"{self.dataset_path}{self.file_name}").get_nii_fov()
        volume_size = fov[0] * fov[1] * fov[2]

        for key in range(1, self.NUM_CLASS):
            volume_list_dict[f'Volume_{self.DICT_CLASS[key]}'] = []

        for mask in self.masks_list:
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
                        clsrf = InstancesFinder(mark_mask, kernel = np.min(mark_mask.shape), num_class = clss)
                        clusters = clsrf.find_clusters()
                        max_size = 0
                        
                        for i in range(len(clusters[:])):
                            cluster_size = len(clusters[i]['coords'])
                            
                            if cluster_size > len(clusters[max_size]['coords']):
                                max_size = i

                            weight_mass_y, weight_mass_x = clusters[max_size]['coords'][len(clusters[max_size]['coords'])//2]

                    ax.annotate(f'{self.SCAR_DICT_CLASS[clss]}', 
                                xy = (weight_mass_x, weight_mass_y), 
                                fontsize = 6, xytext = (weight_mass_x + 5, weight_mass_y + 5), 
                                arrowprops = self.arrowprops,
                                bbox = self.bbox, 
                                color = 'black')

                    ax.plot([weight_mass_x], [weight_mass_y],  marker = ".", color = 'orange')

            for key in range(1, 4): 
                mask_slc[0][key - 1] = key

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
                    relVolume = round((FIBv[slc] / (FIBv[slc] + MYOv[slc] + self.smooth)) * 100, 2)
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
                if class_volume[f"Chunk_{self.DICT_CLASS[key]}"][page][slc] / 1000 > 1:
                    report_title += (
                        f'{self.DICT_CLASS[key]}_vol: {class_volume[f"Chunk_{self.DICT_CLASS[key]}"][page][slc] / 1000} ml, ' )
                
            elif page == None and slc == None:
                report_title += (
                    f'Full {self.DICT_CLASS[key]} volume: {sum(class_volume[f"Volume_{self.DICT_CLASS[key]}"]) / 1000} ml, \n' )

        return report_title

    @property
    def save_pdf(self):
        volume_list_dict = self.get_stats_parameters
        volume_dict_class = self.create_dict_volume_class
        
        for key in range(1, self.NUM_CLASS):
            volume_dict_class[f'Volume_{self.DICT_CLASS[key]}'] = volume_list_dict[f'Volume_{self.DICT_CLASS[key]}']

        num_chunk = len(self.images_list) % self.rows
        chunk_list_images = list(self.divide_chunks(self.images_list, self.rows))
        chunk_list_masks = list(self.divide_chunks(self.masks_list, self.rows))
        chunk_list_orig_masks = list(self.divide_chunks(self.orig_masks_list, self.rows))
        # chunk_list_fib_masks = list(self.divide_chunks(self.fib_masks_list, self.rows))

        for key in range(1, self.NUM_CLASS): 
            volume_dict_class[f'Chunk_{self.DICT_CLASS[key]}'] = list(self.divide_chunks(volume_dict_class[f'Volume_{self.DICT_CLASS[key]}'], self.rows))

        num_pages = len(chunk_list_images)
        pp = PdfPages(f'{self.inference_directory}/{self.file_name}_results.pdf')
        
        for page in range(num_pages):
            images = chunk_list_images[page]
            masks = chunk_list_masks[page]
            orig_masks = chunk_list_orig_masks[page]
            # fib_masks = chunk_list_fib_masks[page]

            images_on_page = len(masks)
            
            if images_on_page > 1:
                num_images = images_on_page
            elif images_on_page == 1:
                num_images = 3

            figure, ax = plt.subplots(nrows = num_images, ncols = 3, figsize = (12, 12))
            colormap = plt.cm.get_cmap('viridis')  # 'plasma' or 'viridis'
            colormap.set_under('k', alpha = .5)

            for slc in range(images_on_page):                    
                image_slc = self.preprocess_matrix(images[slc])
                mask_slc = self.preprocess_matrix(masks[slc])
                orig_mask_slc = self.preprocess_matrix(orig_masks[slc])
                # fib_mask_slc  = self.preprocess_matrix(fib_masks[slc])

                ax[slc, 1], mask_slc = self.add_annotate_class(slc, ax[slc, 1], mask_slc)
                ax[slc, 2], orig_mask_slc = self.add_annotate_class(slc, ax[slc, 2], orig_mask_slc)
                # ax[slc, 2], fib_mask_slc = self.add_annotate_class(slc, ax[slc, 2], fib_mask_slc)

                ax[slc, 0].imshow(image_slc, plt.get_cmap('gray'))

                ax[slc, 1].imshow(image_slc, plt.get_cmap('gray'))
                ax[slc, 1].imshow(mask_slc, alpha = 0.5, interpolation = None, cmap = colormap,  vmin = 0.5)
                ax[slc, 1].contour(mask_slc, alpha = 0.5)

                ax[slc, 2].imshow(image_slc, plt.get_cmap('gray'))
                ax[slc, 2].imshow(orig_mask_slc, alpha = 0.5, interpolation = None, cmap = colormap,  vmin = 0.5)
                ax[slc, 2].contour(orig_mask_slc, alpha = 0.5)

                # ax[slc, 3].imshow(image_slc, plt.get_cmap('gray'))
                # ax[slc, 3].imshow(fib_mask_slc, alpha = 0.5, interpolation = None, cmap = colormap,  vmin = 0.5)
                # ax[slc, 3].contour(fib_mask_slc, alpha = 0.5)

                report_title = ''
                report_title = self.threshold_scar(report_title, page, slc, volume_dict_class)
                report_title = self.write_class_volume(report_title, page, slc, volume_dict_class)

                ax[slc, 1].set_title(report_title, fontsize = 8, fontweight = 'bold', loc = 'right')

                figure.tight_layout()
            pp.savefig(figure)

        report_title = ''
        report_title = self.write_class_volume(report_title, None, None, volume_dict_class)
        report_title = self.threshold_scar(report_title, None, None, volume_dict_class)

        fig = plt.figure(figsize = (8, 8))
        text = fig.text(0.2, 0.7, report_title, ha = 'left', va = 'top', size = 14)

        text.set_path_effects([path_effects.Normal()])
        pp.savefig(fig)
        
        pp.close()


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
                    
                    if (self.old_matrix[coords[0]][coords[1]] != self.extra_symbol):
                        clusterData['coords'].append(coords)
                    else:
                        clusterData['extras'].append(coords)

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

























