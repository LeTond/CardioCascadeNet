 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1
Date: 21-07-2024
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""


from Evaluation.evaluation import *
from Preprocessing.preprocessing import ReadImages
from parameters import MetaParameters
from configuration import *
from Preprocessing.split_dataset import *

from time import time


class Evaluation(MetaParameters):

    def __init__(self):         
        super(MetaParameters, self).__init__()
        self.unet1_eval_dir = self.NEW_UNET1_MASK_PATH
        self.unet2_eval_dir = self.NEW_UNET2_MASK_PATH
        self.unet3_eval_dir = self.NEW_UNET3_MASK_PATH
        self.dataset_path = self.NEW_DATA_PATH
        self.checkpoint = torch.load(f'{self.PROJ_NAME}/{self.DATASET_NAME}_model.pth')

    def nifti_unet1_evaluation(self, file_name):
        create_dir(self.unet1_eval_dir)
        
        checkpoint = self.checkpoint[f'Net_{self.DATASET_NAME}_{self.UNET1_FOLD}']
        neural_model = checkpoint['Model']
        neural_model.load_state_dict(checkpoint['weights'])

        # neural_model = torch.load(f'{self.project_name}/{self.MODEL_NAME}.pth', map_location=torch.device('cpu')).to(device=device)
        images, templates, image_shp, def_coord = GetListImages(file_name, self.unet1_eval_dir, self.dataset_path, unet_type = 'default').nifti_list(None)
        masks_list = PredictionMask(neural_model, self.KERNEL, images, templates, image_shp, def_coord).get_predicted_mask

        # NiftiSaver(masks_list, file_name, self.unet1_eval_dir).save_nifti
        # PdfSaver(file_name, self.dataset_path, self.unet1_eval_dir).save_pdf

        return masks_list

    def nifti_unet2_evaluation(self, file_name, masks_list):
        create_dir(self.unet2_eval_dir)

        checkpoint = self.checkpoint[f'Net_{self.DATASET_NAME}_{self.UNET2_FOLD}']
        neural_model = checkpoint['Model']
        neural_model.load_state_dict(checkpoint['weights'])

        images, templates, image_shp, def_coord = GetListImages(file_name, self.unet1_eval_dir, self.dataset_path, unet_type = 'cropp').nifti_list(masks_list)
        masks_list = PredictionMask(neural_model, self.CROPP_KERNEL, images, templates, image_shp, def_coord).get_predicted_mask
        
        # NiftiSaver(masks_list, file_name, self.unet2_eval_dir).save_nifti
        # PdfSaver(file_name, self.dataset_path, self.unet2_eval_dir).save_pdf
        
        return masks_list

    def nifti_unet3_evaluation(self, file_name, masks_list):
        create_dir(self.unet3_eval_dir)
        
        checkpoint = self.checkpoint[f'Net_{self.DATASET_NAME}_{self.UNET3_FOLD}']
        neural_model = checkpoint['Model']
        neural_model.load_state_dict(checkpoint['weights'])

        images, templates, image_shp, def_coord = GetListImages(file_name, self.unet2_eval_dir, self.dataset_path, unet_type = 'close_cropp').nifti_list(masks_list)
        masks_list = PredictionMask(neural_model, self.CROPP_KERNEL, images, templates, image_shp, def_coord).get_predicted_mask

        NiftiSaver(masks_list, file_name, self.unet3_eval_dir).save_nifti
        PdfSaver(file_name, self.dataset_path, self.unet3_eval_dir).save_pdf

    def dicom_unet1_evaluation(self, file_name, def_coord=None, masks_list=None):
        create_dir(self.unet1_eval_dir)

        checkpoint = self.checkpoint[f'Net_{self.DATASET_NAME}_{self.UNET1_FOLD}']
        neural_model = checkpoint['Model']
        neural_model.load_state_dict(checkpoint['weights'])

        images, image_shp, def_coord = GetListImages(file_name, self.unet1_eval_dir, self.dataset_path, preseg1 = False, preseg2 = False).dicom_array(self.KERNEL, None, None)
        masks_list = PredictionMask(neural_model, self.KERNEL, images, image_shp, def_coord).get_predicted_mask()        

        if self.UNET2:
            DicomSaver(masks_list, file_name, self.unet1_eval_dir).save_dicom_mask()
        else:
            DicomSaver(masks_list, file_name, self.unet1_eval_dir).save_dicom()

        return masks_list

    def dicom_unet2_evaluation(self, file_name, def_coord = None, masks_list = None):
        create_dir(self.unet2_eval_dir)

        checkpoint = self.checkpoint[f'Net_{self.DATASET_NAME}_{self.UNET2_FOLD}']
        neural_model = checkpoint['Model']
        neural_model.load_state_dict(checkpoint['weights'])

        images, image_shp, def_coord = GetListImages(file_name, self.unet1_eval_dir, self.dataset_path, preseg1 = True, preseg2 = False).dicom_array(self.CROPP_KERNEL, def_coord, masks_list)
        masks_list = PredictionMask(neural_model, self.CROPP_KERNEL, images, image_shp, def_coord).get_predicted_mask()

        if self.UNET3:
            DicomSaver(masks_list, file_name, self.unet2_eval_dir).save_dicom_mask()
        else:
            DicomSaver(masks_list, file_name, self.unet2_eval_dir).save_dicom()
        
        return masks_list

    def dicom_unet3_evaluation(self, file_name, def_coord = None, masks_list = None):
        create_dir(self.unet3_eval_dir)

        checkpoint = self.checkpoint[f'Net_{self.DATASET_NAME}_{self.UNET3_FOLD}']
        neural_model = checkpoint['Model']
        neural_model.load_state_dict(checkpoint['weights'])

        images, image_shp, def_coord = GetListImages(file_name, self.unet2_eval_dir, self.dataset_path, preseg1 = True, preseg2 = True).dicom_array(self.CROPP_KERNEL, def_coord, masks_list)
        
        masks_list = PredictionMask(neural_model, self.CROPP_KERNEL, images, image_shp, def_coord).get_predicted_mask()
        
        DicomSaver(masks_list, file_name, self.unet3_eval_dir).save_dicom()

        # return masks_list

    def get_coordination(self, file_name, def_coord = None, masks_list = None):
        images, image_shp, def_coord = GetListImages(file_name, self.unet1_eval_dir, self.dataset_path, preseg1 = True, preseg2 = False).dicom_array(self.CROPP_KERNEL, def_coord, masks_list)

        return def_coord

    def get_coordination_2(self, file_name, def_coord = None, masks_list = None):
        images, image_shp, def_coord = GetListImages(file_name, self.unet2_eval_dir, self.dataset_path, preseg1 = True, preseg2 = True).dicom_array(self.CROPP_KERNEL, def_coord, masks_list)
        
        return def_coord

    def create_dict_subnames(self, subname):
        dict_sub_names = {}
        dict_sub_names[f'Subname_{subname}'] = []

        return dict_sub_names

    def run_process(self):
        # dataset_list = ReadImages(f'{self.DATASET_DIR}{self.DATASET_NAME}_origin_new/').get_dataset_list()
        # dataset_list = ReadImages(f'{self.DATASET_DIR}{self.DATASET_NAME}_origin_new/').get_file_path_list()
        dataset_list = test_list

        ##TODO: should add while Patient.name == FixPatient.name: continiue else: def_coord_list = [] coord_x, coord_y = 0, 0

        unet1_coord_list, unet2_coord_list = [], []
        masks_list_01, masks_list_02 = [], []
        coord_x, coord_y = 0, 0

        subname = 'default'
        dict_sub_names = self.create_dict_subnames(subname)

        for file_name in dataset_list:
            if file_name.endswith('.nii'):
                if self.UNET1 is True:
                    masks_list_01 = self.nifti_unet1_evaluation(file_name)
                    print(f'New subject {file_name} was saved with base U-net1 Model')

                if self.UNET2 is True:
                    masks_list_02 = self.nifti_unet2_evaluation(file_name, masks_list_01)
                    print(f'New subject {file_name} was saved with U-net2 Model')

                if self.UNET3 is True:
                    self.nifti_unet3_evaluation(file_name, masks_list_02)
                    print(f'New subject {file_name} was saved with U-net3 Model')

        for file_name in dataset_list:
            if file_name.endswith('.dcm') and self.UNET1 is True:
                masks_list = self.dicom_unet1_evaluation(file_name)
                masks_list_01.append(masks_list)
                unet1_coord_list.append(self.get_coordination(file_name, None, np.array(masks_list)))
                print(f'New subject {file_name} was saved with base U-net1 Model')

        for file_name in dataset_list:
            for coord in unet1_coord_list:
                coord_x += coord[0]
                coord_y += coord[1]

            coord_x = coord_x // len(unet1_coord_list)
            coord_y = coord_y // len(unet1_coord_list)
            unet1_coord = [coord_x, coord_y]

            if file_name.endswith('.dcm') and self.UNET2 is True:
                masks_list = self.dicom_unet2_evaluation(file_name, unet1_coord, np.array(masks_list_01))
                masks_list_02.append(masks_list)
                unet2_coord_list.append(self.get_coordination_2(file_name, unet1_coord, np.array(masks_list)))
                print(f'New subject {file_name} was saved with U-net2 Model')

        coord_x, coord_y = 0, 0

        for file_name in dataset_list:
            for coord in unet2_coord_list:
                coord_x += coord[0]
                coord_y += coord[1]

            coord_x = coord_x // len(unet2_coord_list)
            coord_y = coord_y // len(unet2_coord_list)
            unet2_coord = [coord_x, coord_y]

            if file_name.endswith('.dcm') and self.UNET3 is True:
                self.dicom_unet3_evaluation(file_name, unet2_coord, np.array(masks_list_02))
                print(f'New subject {file_name} was saved with U-net3 Model')

            
        for file_name in dataset_list:
            if file_name.endswith('.dcm') and self.UNET3 is True:
                masks_01 = self.dicom_unet1_evaluation(file_name)
                masks_02 = self.dicom_unet2_evaluation(file_name, unet1_coord, np.array(masks_01))
                self.dicom_unet3_evaluation(file_name, unet2_coord, np.array(masks_02))



if __name__ == "__main__":
    Evaluation().run_process()
