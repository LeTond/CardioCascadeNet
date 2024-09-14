 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.2
Date: 03-09-2024
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""


from Inference.inference import *
from Preprocessing.preprocessing import ReadImages
from configuration import *
from Preprocessing.split_dataset import *

from time import time


class Inference(MetaParameters):
    def __init__(self):         
        super(MetaParameters, self).__init__()
        self.unet1_infer_dir = self.NEW_UNET1_MASK_PATH
        self.unet2_infer_dir = self.NEW_UNET2_MASK_PATH
        self.unet3_infer_dir = self.NEW_UNET3_MASK_PATH
        self.unet4_infer_dir = self.NEW_UNET4_MASK_PATH
        self.unet5_infer_dir = self.NEW_UNET5_MASK_PATH
        self.dataset_path = self.NEW_DATA_PATH
        self.checkpoint = torch.load(f'{self.PROJ_NAME}/{self.DATASET_NAME}_model.pth')

    def nifti_unet1_inference(self, file_name, masks_list = None):
        fdwr.create_dir(project_name = self.unet1_infer_dir)
        
        checkpoint = self.checkpoint[f'Net_{self.DATASET_NAME}_{self.UNET1_FOLD}']
        neural_model = checkpoint['Model']
        neural_model.load_state_dict(checkpoint['weights'])

        images, templates, image_shp, def_coord = \
        GetListImages(file_name, self.unet1_infer_dir, self.dataset_path, unet_type = 'default').nifti_list(masks_list)
        
        masks_list = PredictionMask(neural_model, images, templates, image_shp, def_coord, unet_type = 'default').get_predicted_mask

        NiftiSaver(masks_list, file_name, self.unet1_infer_dir).save_nifti
        # PdfSaver(file_name, self.dataset_path, self.unet1_infer_dir).save_pdf

        return masks_list

    def nifti_unet2_inference(self, file_name, masks_list):
        fdwr.create_dir(project_name = self.unet2_infer_dir)

        checkpoint = self.checkpoint[f'Net_{self.DATASET_NAME}_{self.UNET2_FOLD}']
        neural_model = checkpoint['Model']
        neural_model.load_state_dict(checkpoint['weights'])

        images, templates, image_shp, def_coord = \
        GetListImages(file_name, self.unet1_infer_dir, self.dataset_path, unet_type = 'cropp').nifti_list(masks_list)
        
        masks_list = PredictionMask(neural_model, images, templates, image_shp, def_coord, unet_type = 'cropp').get_predicted_mask
        
        NiftiSaver(masks_list, file_name, self.unet2_infer_dir).save_nifti
        # PdfSaver(file_name, self.dataset_path, self.unet2_infer_dir).save_pdf
        
        return masks_list

    def nifti_unet3_inference(self, file_name, masks_list):
        fdwr.create_dir(project_name = self.unet3_infer_dir)
        
        checkpoint = self.checkpoint[f'Net_{self.DATASET_NAME}_{self.UNET3_FOLD}']
        neural_model = checkpoint['Model']
        neural_model.load_state_dict(checkpoint['weights'])

        images, templates, image_shp, def_coord = \
        GetListImages(file_name, self.unet2_infer_dir, self.dataset_path, unet_type = 'close_cropp').nifti_list(masks_list)
        
        masks_list = PredictionMask(neural_model, images, templates, image_shp, def_coord, unet_type = 'close_cropp').get_predicted_mask

        NiftiSaver(masks_list, file_name, self.unet3_infer_dir).save_nifti
        # PdfSaver(file_name, self.dataset_path, self.unet3_infer_dir).save_pdf

        return masks_list

    def nifti_unet4_inference(self, file_name, masks_list = None):
        fdwr.create_dir(project_name = self.unet4_infer_dir)
        
        checkpoint = self.checkpoint[f'Net_{self.DATASET_NAME}_{self.UNET4_FOLD}']
        neural_model = checkpoint['Model']
        neural_model.load_state_dict(checkpoint['weights'])

        images, templates, image_shp, def_coord = \
        GetListImages(file_name, self.unet3_infer_dir, self.dataset_path, unet_type = 'cropp', mask_type = 'myo_level').nifti_list(masks_list)
        
        masks_list = PredictionMask(neural_model, images, templates, image_shp, def_coord, unet_type = 'cropp').get_predicted_mask

        NiftiSaver(masks_list, file_name, self.unet4_infer_dir).save_nifti
        # PdfSaver(file_name, self.dataset_path, self.unet4_infer_dir).save_pdf

        return masks_list

    def nifti_unet5_inference(self, file_name, masks_list = None):
        fdwr.create_dir(project_name = self.unet5_infer_dir)
        
        checkpoint = self.checkpoint[f'Net_{self.DATASET_NAME}_{self.UNET5_FOLD}']
        neural_model = checkpoint['Model']
        neural_model.load_state_dict(checkpoint['weights'])

        images, templates, image_shp, def_coord = \
        GetListImages(file_name, self.unet4_infer_dir, self.dataset_path, unet_type = 'cropp', mask_type = 'infer_bull_level').nifti_list(masks_list)
        
        masks_list = PredictionMask(neural_model, images, templates, image_shp, def_coord, unet_type = 'cropp').get_predicted_mask

        NiftiSaver(masks_list, file_name, self.unet5_infer_dir).save_nifti
        PdfSaver(file_name, self.dataset_path, self.unet5_infer_dir).save_pdf

    def dicom_unet1_inference(self, file_name, def_coord = None):
        fdwr.create_dir(project_name = self.unet1_infer_dir)

        checkpoint = self.checkpoint[f'Net_{self.DATASET_NAME}_{self.UNET1_FOLD}']
        neural_model = checkpoint['Model']
        neural_model.load_state_dict(checkpoint['weights'])

        images, templates, image_shp, def_coord = \
        GetListImages(file_name, self.unet1_infer_dir, self.dataset_path, unet_type = 'default').dicom_array(def_coord, None)
        
        masks_list = PredictionMask(neural_model, images, templates, image_shp, def_coord, unet_type = 'default').get_predicted_mask      

        if self.UNET2:
            DicomSaver(masks_list, file_name, self.unet1_infer_dir).save_dicom_mask()
        else:
            DicomSaver(masks_list, file_name, self.unet1_infer_dir).save_dicom()

        return masks_list

    def dicom_unet2_inference(self, file_name, def_coord = None, masks_list = None):
        fdwr.create_dir(project_name = self.unet2_infer_dir)

        checkpoint = self.checkpoint[f'Net_{self.DATASET_NAME}_{self.UNET2_FOLD}']
        neural_model = checkpoint['Model']
        neural_model.load_state_dict(checkpoint['weights'])

        images, templates, image_shp, def_coord = \
        GetListImages(file_name, self.unet1_infer_dir, self.dataset_path, unet_type = 'cropp').dicom_array(def_coord, masks_list)
        
        masks_list = PredictionMask(neural_model, images, templates, image_shp, def_coord, unet_type = 'cropp').get_predicted_mask        

        if self.UNET3:
            DicomSaver(masks_list, file_name, self.unet2_infer_dir).save_dicom_mask()
        else:
            DicomSaver(masks_list, file_name, self.unet2_infer_dir).save_dicom()
        
        return masks_list

    def dicom_unet3_inference(self, file_name, def_coord = None, masks_list = None):
        fdwr.create_dir(project_name = self.unet3_infer_dir)

        checkpoint = self.checkpoint[f'Net_{self.DATASET_NAME}_{self.UNET3_FOLD}']
        neural_model = checkpoint['Model']
        neural_model.load_state_dict(checkpoint['weights'])

        images, templates, image_shp, def_coord = \
        GetListImages(file_name, self.unet2_infer_dir, self.dataset_path, unet_type = 'close_cropp').dicom_array(def_coord, masks_list)
        
        masks_list = PredictionMask(neural_model, images, templates, image_shp, def_coord, unet_type = 'close_cropp').get_predicted_mask
        
        DicomSaver(masks_list, file_name, self.unet3_infer_dir).save_dicom()

        return masks_list

    def get_default_coord(self, file_name, def_coord = None, masks_list = None, unet_type = None):
        images, templates, image_shp, def_coord = \
        GetListImages(file_name, self.unet1_infer_dir, self.dataset_path, unet_type).dicom_array(def_coord, masks_list)

        return def_coord

    def get_cropp_coord(self, file_name, def_coord = None, masks_list = None, unet_type = None):
        images, templates, image_shp, def_coord = \
        GetListImages(file_name, self.unet2_infer_dir, self.dataset_path, unet_type).dicom_array(def_coord, masks_list)
        
        return def_coord

    def create_dict_subnames(self, subname):
        dict_sub_names = {}
        dict_sub_names[f'Subname_{subname}'] = []

        return dict_sub_names

    def rewrite_weights(self, from_model, to_model, unet_model):
        checkpoint1 = torch.load(f'./Results/{from_model}/{from_model}_model.pth')
        checkpoint1 = checkpoint1[f'Net_{from_model}_{unet_model}']

        model = checkpoint1['Model']
        model.load_state_dict(checkpoint1['weights'])  

        checkpoint = torch.load(f'./Results/{to_model}/{to_model}_model.pth')
        checkpoint[f'Net_{to_model}_{unet_model}'] = {'Model': checkpoint1['Model'], 'weights': checkpoint1['Model'].state_dict()}

        torch.save(
            checkpoint,
            f'./Results/{to_model}/{to_model}_model.pth')

    def run_process(self):
        # dataset_list = ReadImages(f'{self.DATASET_DIR}{self.DATASET_NAME}_origin_new/').get_dataset_list()
        # dataset_list = ReadImages(f'{self.DATASET_DIR}{self.DATASET_NAME}_origin_new/').get_file_path_list()
        jsnlst = JsonFoldList()
        dataset_list = jsnlst.load_dataset_list('test_list')
        jsnlst.pprint('test_list')

        ##TODO: should add while Patient.name == FixPatient.name: continiue else: def_coord_list = [] coord_x, coord_y = 0, 0
        unet1_coord_list, unet2_coord_list = [], []
        masks_list_01, masks_list_02 = [], []
        coord_x, coord_y = 0, 0

        subname = 'default'
        dict_sub_names = self.create_dict_subnames(subname)

        self.rewrite_weights('BULLEYE', 'ALMAZ', 'Unet4_Fold_full/')
        self.rewrite_weights('BULLEYE', 'ALMAZ', 'Unet5_Fold_full/')

        for file_name in dataset_list:
            if file_name.endswith('.nii'):
                if self.UNET1 is True:
                    masks_list_01 = self.nifti_unet1_inference(file_name)
                    print(f'New subject {file_name} was saved with base U-net1 Model')

                if self.UNET2 is True:
                    masks_list_02 = self.nifti_unet2_inference(file_name, masks_list_01)
                    print(f'New subject {file_name} was saved with U-net2 Model')

                if self.UNET3 is True:
                    masks_list_03 = self.nifti_unet3_inference(file_name, masks_list_02)
                    print(f'New subject {file_name} was saved with U-net3 Model')

                if self.UNET4 is True:
                    masks_list_04 = self.nifti_unet4_inference(file_name, masks_list_03)
                    print(f'New subject {file_name} was saved with U-net4 Model')

                if self.UNET5 is True:
                    self.nifti_unet5_inference(file_name, masks_list_04)
                    print(f'New subject {file_name} was saved with U-net5 Model')


        for file_name in dataset_list:
            if file_name.endswith('.dcm') and self.UNET1 is True:
                masks_list = self.dicom_unet1_inference(file_name)
                masks_list_01.append(masks_list)
                unet1_coord_list.append(self.get_default_coord(file_name, None, np.array(masks_list), unet_type = 'default'))
                print(f'New subject {file_name} was saved with base U-net1 Model')

        for file_name in dataset_list:
            for coord in unet1_coord_list:
                coord_x += coord[0]
                coord_y += coord[1]

            coord_x = coord_x // len(unet1_coord_list)
            coord_y = coord_y // len(unet1_coord_list)
            unet1_coord = [coord_x, coord_y]

            if file_name.endswith('.dcm') and self.UNET2 is True:
                masks_list = self.dicom_unet2_inference(file_name, unet1_coord, np.array(masks_list_01))
                masks_list_02.append(masks_list)
                unet2_coord_list.append(self.get_cropp_coord(file_name, unet1_coord, np.array(masks_list), unet_type = 'cropp'))
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
                self.dicom_unet3_inference(file_name, unet2_coord, np.array(masks_list_02))
                print(f'New subject {file_name} was saved with U-net3 Model')

        for file_name in dataset_list:
            if file_name.endswith('.dcm') and self.UNET3 is True:
                masks_01 = self.dicom_unet1_inference(file_name)
                masks_02 = self.dicom_unet2_inference(file_name, unet1_coord, np.array(masks_01))
                self.dicom_unet3_inference(file_name, unet2_coord, np.array(masks_02))



if __name__ == "__main__":
    Inference().run_process()
