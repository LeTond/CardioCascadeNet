 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.6
Date: 10-02-2026
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""


import time
import torch

import CardioCascadeNet



class MeasureTime:
    def __init__(self):
        print("Отсчет времени выполнения включен")

    def __call__(self, func):
        def wrapper(*args, **kwargs):
        
            start = time.time()
            func(*args, **kwargs)
            end = time.time()
            print(f"Время выполнения сегментации: {end - start} с")
        
        return wrapper


class InferenceRun(CardioCascadeNet.MetaParameters):
    def __init__(self):         
        super(CardioCascadeNet.MetaParameters, self).__init__()
        
        self.fdwr = CardioCascadeNet.FileDirectoryWorker()
        self.dataset_path = self.NEW_DATA_PATH
        self.checkpoint = torch.load(f'{self.PROJ_NAME}/{self.DATASET_NAME}_model.pth')
        # self.checkpoint = torch.load(f'{self.PROJ_NAME}/{self.DATASET_NAME}_model.pth', map_location = torch.device('cpu'))

    def get_images(self, file_name, masks_list, unet_type, mask_type, file_type):
        if file_type == 'nifti_type':
            images, templates, image_shp, def_coord = \
            CardioCascadeNet.PredictListImages(file_name, self.dataset_path, unet_type, mask_type).nifti_list(masks_list)

        elif file_type == 'dicom_type':            
            images, templates, image_shp, def_coord = \
            CardioCascadeNet.PredictListImages(file_name, self.dataset_path, unet_type, mask_type).dicom_array(masks_list)

        return images, templates, image_shp, def_coord

    def model_inference(self, **kwargs):
        file_dir, file_name, masks_list, model_fold, unet_type, mask_type, file_type, pdf_flag = \
        kwargs['file_dir'], kwargs['file_name'], kwargs['masks_list'], kwargs['model_fold'], \
        kwargs['unet_type'], kwargs['mask_type'], kwargs['file_type'], kwargs['pdf_flag']

        self.fdwr.create_dir(project_name = file_dir)
        
        checkpoint = self.checkpoint[f"Net_{self.DATASET_NAME}_{model_fold}"]
        neural_model = checkpoint['Model']
        neural_model.load_state_dict(checkpoint['weights'])

        images, templates, image_shp, def_coord = self.get_images(
            file_name, masks_list, unet_type, mask_type, file_type)
        
        masks_list = CardioCascadeNet.PredictionMask(
            neural_model, images, templates, image_shp, def_coord, unet_type = unet_type).get_predicted_mask

        if mask_type == 'infer_bull_level':
            try:
                masks_list = CardioCascadeNet.MaskPostprocessing(file_name = file_name, masks_list = masks_list, mask_type = mask_type).check_bull_apex

            except Exception as e:
                print(e)
            pass

        if file_type == 'nifti_type':
            CardioCascadeNet.NiftiSaver(masks_list, file_name, file_dir).save_nifti

            if pdf_flag is True:
                CardioCascadeNet.PdfSaver(file_name, self.dataset_path, file_dir).save_pdf

        if file_type == 'dicom_type':
            for slc in range(len(file_name)):
                if model_fold == self.UNET3_FOLD or model_fold == self.UNET4_FOLD:
                    CardioCascadeNet.DicomSaver(masks_list[:, :, slc], file_name[slc], file_dir).save_dicom_mask()
                else:
                    CardioCascadeNet.DicomSaver(masks_list[:, :, slc], file_name[slc], file_dir).save_dicom()

        return masks_list

    def dict_subnames(self, dict_sub_names, subname = 'default', file_name = ''):
        try:
            dict_sub_names[subname].append(f"{file_name}")
        except:
            dict_sub_names[subname] = [f"{file_name}"]

        return dict_sub_names

    def run_process(self):
        jsnlst = CardioCascadeNet.JsonFoldList()
        jsnlst.create_folds_list

        dataset_list = jsnlst.load_dataset_list('test_list')
        # dataset_list = CardioCascadeNet.ReadImages(f'{self.DATASET_DIR}{self.DATASET_NAME}_images_new/').get_dataset_list()
        # dataset_list = CardioCascadeNet.ReadImages(f'{self.DATASET_DIR}{self.DATASET_NAME}_images_new/').get_file_path_list()

        jsnlst.pprint('test_list')
        # jsnlst.pprint('train_list')
        # print(dataset_list)
        print('\nINFERENCE RUN\n')

        ###########################################################################################################
        ##  Nifti file inference (.nii.gz)
        ###########################################################################################################
        file_type = 'nifti_type'
        try:
            for file_name in dataset_list:
                if file_name.endswith('.nii.gz'):
                    if self.UNET1 is True:
                        masks_list_01 = self.model_inference(
                            file_dir = self.NEW_UNET1_MASKS_PATH, file_name = file_name, 
                            masks_list = None, model_fold = self.UNET1_FOLD, 
                            unet_type = 'default', mask_type = None, file_type = file_type, pdf_flag = self.PDF_FLAG_UNET1)
                            
                    if self.UNET2 is True:
                        masks_list_02 = self.model_inference(
                            file_dir = self.NEW_UNET2_MASKS_PATH, file_name = file_name, 
                            masks_list = masks_list_01, model_fold = self.UNET2_FOLD, 
                            unet_type = 'cropp', mask_type = None, file_type = file_type, pdf_flag = self.PDF_FLAG_UNET2)

                    if self.UNET3 is True:
                        masks_list_03 = self.model_inference(
                            file_dir = self.NEW_UNET3_MASKS_PATH, file_name = file_name, 
                            masks_list = masks_list_02, model_fold = self.UNET3_FOLD, 
                            unet_type = 'close_cropp', mask_type = None, file_type = file_type, pdf_flag = self.PDF_FLAG_UNET3)

                    if self.UNET4 is True:
                        masks_list_04 = self.model_inference(
                            file_dir = self.NEW_UNET4_MASKS_PATH, file_name = file_name, 
                            masks_list = masks_list_03, model_fold = self.UNET4_FOLD, 
                            unet_type = 'cropp', mask_type = 'lv_level', file_type = file_type, pdf_flag = self.PDF_FLAG_UNET4)

                    if self.UNET5 is True:
                        self.model_inference(
                            file_dir = self.NEW_UNET5_MASKS_PATH, file_name = file_name, 
                            masks_list = masks_list_04, model_fold = self.UNET5_FOLD, 
                            unet_type = 'cropp', mask_type = 'infer_bull_level', file_type = file_type, pdf_flag = self.PDF_FLAG_UNET5)

                    print(f'New subject {file_name} was saved')

        except Exception as e:
            print(e)
            pass

        ###########################################################################################################
        ##  DICOM file inference (.dcm)
        ###########################################################################################################
        ##TODO: DICOM inference work only if get mask info from predicted and saved mask into preview directory
        ##TODO: should add while Patient.name == FixPatient.name: continiue else: def_coord_list = [] coord_x, coord_y = 0, 0
        ##TODO: It should be union into one HxWxN matrix 
        dataset_list = CardioCascadeNet.ReadImages(f'{self.DATASET_DIR}{self.DATASET_NAME}_images_new/').get_file_path_list()
        dict_subnames = {}

        for file_name in dataset_list:
            if file_name.endswith('.dcm'):                
                patient_name = CardioCascadeNet.ReadImages(f'{file_name}').get_dcm_name()
                dict_subnames = self.dict_subnames(dict_subnames, patient_name, file_name)
        
        print(dict_subnames.keys())

        file_type = 'dicom_type'

        try: 
            for key_name in dict_subnames.keys():
                if self.UNET1 is True:
                    masks_list_01 = self.model_inference(
                        file_dir = self.NEW_UNET1_MASKS_PATH, file_name = dict_subnames[key_name], 
                        masks_list = None, model_fold = self.UNET1_FOLD, 
                        unet_type = 'default', mask_type = None, file_type = file_type, pdf_flag = self.PDF_FLAG_UNET1)

                if self.UNET2 is True:
                    masks_list_02 = self.model_inference(
                        file_dir = self.NEW_UNET2_MASKS_PATH, file_name = dict_subnames[key_name], 
                        masks_list = masks_list_01, model_fold = self.UNET2_FOLD, 
                        unet_type = 'cropp', mask_type = None, file_type = file_type, pdf_flag = self.PDF_FLAG_UNET2)

                if self.UNET3 is True:
                    masks_list_03 = self.model_inference(
                        file_dir = self.NEW_UNET3_MASKS_PATH, file_name = dict_subnames[key_name], 
                        masks_list = masks_list_02, model_fold = self.UNET3_FOLD, 
                        unet_type = 'close_cropp', mask_type = None, file_type = file_type, pdf_flag = self.PDF_FLAG_UNET3)

                if self.UNET4 is True:
                    masks_list_04 = self.model_inference(
                        file_dir = self.NEW_UNET4_MASKS_PATH, file_name = dict_subnames[key_name], 
                        masks_list = masks_list_03, model_fold = self.UNET4_FOLD, 
                        unet_type = 'cropp', mask_type = 'lv_level', file_type = file_type, pdf_flag = self.PDF_FLAG_UNET4)

                if self.UNET5 is True:
                    self.model_inference(
                        file_dir = self.NEW_UNET5_MASKS_PATH, file_name = dict_subnames[key_name], 
                        masks_list = masks_list_04, model_fold = self.UNET5_FOLD, 
                        unet_type = 'cropp', mask_type = 'infer_bull_level', file_type = file_type, pdf_flag = self.PDF_FLAG_UNET5)
                    
                print(f'New subject {file_name} was saved')

        except ZeroDivisionError:
            pass
