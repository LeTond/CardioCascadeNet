 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.4
Date: 04-05-2025
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

    def get_origin_images(self, file_name, masks_list, unet_type, mask_type, nifti_type, dicom_type):
        if nifti_type == True:
            images, templates, image_shp, def_coord = \
            CardioCascadeNet.PredictListImages(file_name, self.dataset_path, unet_type, mask_type).nifti_list(masks_list)

        elif dicom_type == True:            
            images, templates, image_shp, def_coord = \
            CardioCascadeNet.PredictListImages(file_name, self.dataset_path, unet_type, mask_type).dicom_array(masks_list)

        return images, templates, image_shp, def_coord

    def model_inference(self, **kwargs):
        file_dir, file_name, masks_list, model_fold, \
        unet_type, mask_type, nifti_type, dicom_type, pdf_flag = \
        kwargs['file_dir'], kwargs['file_name'], kwargs['masks_list'], kwargs['model_fold'], \
        kwargs['unet_type'], kwargs['mask_type'], kwargs['nifti_type'], kwargs['dicom_type'], kwargs['pdf_flag']

        self.fdwr.create_dir(project_name = file_dir)
        
        checkpoint = self.checkpoint[f"Net_{self.DATASET_NAME}_{model_fold}"]
        neural_model = checkpoint['Model']
        neural_model.load_state_dict(checkpoint['weights'])

        images, templates, image_shp, def_coord = self.get_origin_images(
            file_name, masks_list, unet_type, mask_type, nifti_type, dicom_type)
        
        masks_list = CardioCascadeNet.PredictionMask(
            neural_model, images, templates, image_shp, def_coord, unet_type = unet_type).get_predicted_mask

        if mask_type == 'infer_bull_level':
            masks_list = CardioCascadeNet.MaskPostprocessing(file_name = file_name, masks_list = masks_list, mask_type = mask_type).check_bull_apex

        if nifti_type:
            CardioCascadeNet.NiftiSaver(masks_list, file_name, file_dir).save_nifti

            if pdf_flag is True:
                CardioCascadeNet.PdfSaver(file_name, self.dataset_path, file_dir).save_pdf

        if dicom_type:
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
        # dataset_list = jsnlst.load_dataset_list('train_list')
        # dataset_list = jsnlst.load_dataset_list('valid_list')
        # jsnlst.pprint('test_list')
        print(dataset_list)

        # dataset_list = CardioCascadeNet.ReadImages(f'{self.DATASET_DIR}{self.DATASET_NAME}_origin_new/').get_dataset_list()
        # dataset_list = CardioCascadeNet.ReadImages(f'{self.DATASET_DIR}{self.DATASET_NAME}_origin_new/').get_file_path_list()

        ###########################################################################################################
        ##  Nifti file inference (.nii)
        ###########################################################################################################
        for file_name in dataset_list:
            if file_name.endswith('.nii'):
                if self.UNET1 is True:
                    masks_list_01 = self.model_inference(
                        file_dir = self.NEW_UNET1_MASK_PATH, file_name = file_name, masks_list = None, 
                        model_fold = self.UNET1_FOLD, unet_type = 'default', mask_type = None, 
                        pdf_flag = False, nifti_type = True, dicom_type = False)

                if self.UNET2 is True:
                    masks_list_02 = self.model_inference(
                        file_dir = self.NEW_UNET2_MASK_PATH, file_name = file_name, masks_list = masks_list_01, 
                        model_fold = self.UNET2_FOLD, unet_type = 'cropp', mask_type = None, 
                        pdf_flag = False, nifti_type = True, dicom_type = False)

                if self.UNET3 is True:
                    masks_list_03 = self.model_inference(
                        file_dir = self.NEW_UNET3_MASK_PATH, file_name = file_name, masks_list = masks_list_02, 
                        model_fold = self.UNET3_FOLD, unet_type = 'close_cropp', mask_type = None, 
                        pdf_flag = False, nifti_type = True, dicom_type = False)

                if self.UNET4 is True:
                    masks_list_04 = self.model_inference(
                        file_dir = self.NEW_UNET4_MASK_PATH, file_name = file_name, masks_list = masks_list_03, 
                        model_fold = self.UNET4_FOLD, unet_type = 'cropp', mask_type = 'lv_level', 
                        pdf_flag = False, nifti_type = True, dicom_type = False)

                if self.UNET5 is True:
                    self.model_inference(
                        file_dir = self.NEW_UNET5_MASK_PATH, file_name = file_name, masks_list = masks_list_04, 
                        model_fold = self.UNET5_FOLD, unet_type = 'cropp', mask_type = 'infer_bull_level', 
                        pdf_flag = True, nifti_type = True, dicom_type = False)

                print(f'New subject {file_name} was saved')

        ###########################################################################################################
        ##  DICOM file inference (.dcm)
        ###########################################################################################################
        ##TODO: DICOM inference work only if get mask info from predicted and saved mask into preview directory
        ##TODO: should add while Patient.name == FixPatient.name: continiue else: def_coord_list = [] coord_x, coord_y = 0, 0
        ##TODO: It should be union into one HxWxN matrix 
        dataset_list = CardioCascadeNet.ReadImages(f'{self.DATASET_DIR}{self.DATASET_NAME}_origin_new/').get_file_path_list()

        dict_subnames = {}

        for file_name in dataset_list:
            if file_name.endswith('.dcm'):                
                patient_name = CardioCascadeNet.ReadImages(f'{file_name}').get_dcm_name()
                dict_subnames = self.dict_subnames(dict_subnames, patient_name, file_name)

        # file_dir = [self.NEW_UNET1_MASK_PATH, self.NEW_UNET2_MASK_PATH, 
                    # self.NEW_UNET3_MASK_PATH, self.NEW_UNET4_MASK_PATH, self.NEW_UNET5_MASK_PATH]
        # masks_list = [None, ]
        # model_fold = [self.UNET1_FOLD, self.UNET2_FOLD, self.UNET3_FOLD, self.UNET4_FOLD, self.UNET5_FOLD] 
        # unet_type = ['default', 'cropp', 'close_cropp', 'cropp', 'cropp']
        # mask_type = [None, None, None, 'lv_level', 'infer_bull_level']
        # pdf_flag = [False, False, False, False, False]
        # nifti_type = [False, False, False, False, False]
        # dicom_type = [True, True, True, True, True]
        
        print(dict_subnames.keys())

        try: 
            for key_name in dict_subnames.keys():
                if self.UNET1 is True:
                    masks_list_01 = self.model_inference(
                        file_dir = self.NEW_UNET1_MASK_PATH, file_name = dict_subnames[key_name], masks_list = None, 
                        model_fold = self.UNET1_FOLD, unet_type = 'default', mask_type = None, 
                        pdf_flag = False, nifti_type = False, dicom_type = True)

                if self.UNET2 is True:
                    masks_list_02 = self.model_inference(
                        file_dir = self.NEW_UNET2_MASK_PATH, file_name = dict_subnames[key_name], masks_list = masks_list_01, 
                        model_fold = self.UNET2_FOLD, unet_type = 'cropp', mask_type = None, 
                        pdf_flag = False, nifti_type = False, dicom_type = True)

                if self.UNET3 is True:
                    masks_list_03 = self.model_inference(
                        file_dir = self.NEW_UNET3_MASK_PATH, file_name = dict_subnames[key_name], masks_list = masks_list_02, 
                        model_fold = self.UNET3_FOLD, unet_type = 'close_cropp', mask_type = None, 
                        pdf_flag = False, nifti_type = False, dicom_type = True)

                if self.UNET4 is True:
                    masks_list_04 = self.model_inference(
                        file_dir = self.NEW_UNET4_MASK_PATH, file_name = dict_subnames[key_name], masks_list = masks_list_03, 
                        model_fold = self.UNET4_FOLD, unet_type = 'cropp', mask_type = 'lv_level', 
                        pdf_flag = False, nifti_type = False, dicom_type = True)

                if self.UNET5 is True:
                    self.model_inference(
                        file_dir = self.NEW_UNET5_MASK_PATH, file_name = dict_subnames[key_name], masks_list = masks_list_04, 
                        model_fold = self.UNET5_FOLD, unet_type = 'cropp', mask_type = 'infer_bull_level', 
                        pdf_flag = False, nifti_type = False, dicom_type = True)
                    
                print(f'New subject {file_name} was saved')

        except ZeroDivisionError:
            pass


if __name__ == "__main__":
    InferenceRun().run_process()
