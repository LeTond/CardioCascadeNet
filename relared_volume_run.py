 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.3
Date: 03-01-2025
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""


from Validation.comparing import *
from Preprocessing.split_dataset import *
from Postprocessing.postprocessing import InstancesFinder

jsnlst = JsonFoldList()
dataset_list = jsnlst.load_dataset_list('test_list')
# dataset_list = jsnlst.load_dataset_list('train_list')
# dataset_list = jsnlst.load_dataset_list('valid_list')
jsnlst.pprint('test_list')
# jsnlst.pprint('train_list')
# jsnlst.pprint('valid_list')


class CountRelVolume(MetaParameters):
    def __init__(self, path_to_label: str, path_to_prediction: str):
        super(MetaParameters, self).__init__()

        self.path_to_label = path_to_label
        self.path_to_prediction = path_to_prediction

        self.fibrosis = self.load_matrix(self.path_to_label, False)
        self.bulleye = self.load_matrix(self.path_to_prediction, True)
        self.length = self.bulleye.shape[-1]

        self.smooth = 1e-5

    def sub_name(self):
        name = self.path_to_label.split('/')[-1]
        name = name.rstrip('.nii')
        
        return name

    @staticmethod
    def load_matrix(path_to_matrix:str, cond:bool):
        matrix = nib.load(path_to_matrix)
        matrix = np.array(matrix.dataobj)

        if np.max(matrix) > 17: 
            print("ERRRROOOR")

        return matrix

    def rel_volume(self):
        num_class = len(self.DICT_CLASS)
        dictionary = {}

        for key in range(1, num_class):
            dictionary[f'RelVol_{self.DICT_CLASS[key]}'] = 0

        # for slc in range(self.length):
        fibrosis = self.fibrosis[:,:,:]
        bulleye = self.bulleye[:,:,:]
        
        for key in range(1, num_class):
            fibrosis_ = fibrosis.copy()
            bulleye_ = bulleye.copy()

            bulleye_[bulleye_!=key] = 0
            bulleye_[bulleye_==key] = 1
            fibrosis_ = fibrosis_ * bulleye_

            fib = fibrosis_[fibrosis_==3]
            myo = fibrosis_[fibrosis_==2]

            fib[fib!=3] = 0
            fib[fib!=0] = 1

            myo[myo!=2] = 0
            myo[myo!=0] = 1

            fib = fib.sum().item()
            myo = myo.sum().item()

            rel_volume = (fib) / (fib + myo + self.smooth) * 100
            dictionary[f'RelVol_{self.DICT_CLASS[key]}'] = seg_check
        
        return dictionary

    def check_transmural(self):
        """
        Example: Sub22
        :return: {'Seg 01': ['-'], 'Seg 02': ['3'], 'Seg 03': ['3'], 'Seg 04': ['3'], 'Seg 05': ['3'], 'Seg 06': ['-'], 
        'Seg 07': ['2', '3'], 'Seg 08': ['2', '3'], 'Seg 09': ['3'], 'Seg 10': ['-'], 'Seg 11': ['3'], 'Seg 12': ['2', '3'], 
        'Seg 13': ['3'], 'Seg 14': ['3'], 'Seg 15': ['3'], 'Seg 16': ['3'], 
        'Seg 17': ['-']}
        Where are:  '-' - segment without fibrosis; 
                    '0' - None
                    '1' - subEndo fibrosis
                    '2' - subEpi fibrosis
                    '3' - Transmural fibrosis
                    '4' - Intramural fibrosis
        """

        num_class = len(self.DICT_CLASS)
        dictionary = {}

        for key in range(1, num_class):
            dictionary[f'{self.DICT_CLASS[key]}'] = []

        for slc in range(self.length):
            fibrosis = self.fibrosis[:, :, slc]
            bulleye = self.bulleye[:, :, slc]
        
            for key in range(1, num_class):
                fibrosis1 = fibrosis.copy()
                fibrosis2 = fibrosis.copy()
                
                bulleye1 = bulleye.copy()
                bulleye2 = bulleye.copy()
                
                bulleye1[bulleye1 != key] = 0
                bulleye1[bulleye1 == key] = 1

                bulleye2[bulleye2 != key] = 1
                bulleye2[bulleye2 == key] = 0

                fibrosis_new2 = fibrosis2 * bulleye2
                fibrosis_new2[fibrosis_new2 == 2] = 97
                fibrosis_new2[fibrosis_new2 == 3] = 98

                fibrosis_new1 = fibrosis1 * bulleye1
                fibrosis_new = fibrosis_new1 + fibrosis_new2

                seg_check = InstancesFinder(fibrosis_new, kernel = 144, num_class = 3).transcheck()
                dictionary[f'{self.DICT_CLASS[key]}'] += ([i for i in seg_check])
        
        for key in range(1, num_class):
            dictionary[f'{self.DICT_CLASS[key]}'] = list(set(dictionary[f'{self.DICT_CLASS[key]}']))
            if len(dictionary[f'{self.DICT_CLASS[key]}']) > 1:
                dictionary[f'{self.DICT_CLASS[key]}'].remove('-')

        return dictionary

    def print(self):
        print(self.sub_name())
        # print(self.rel_volume())
        print(self.check_transmural())
        print()


if __name__ == "__main__":
    # dataset_list = ['Sub01.nii', 'Sub02.nii', 'Sub03.nii', 'Sub04.nii', 'Sub05.nii', 'Sub06.nii', 'Sub07.nii', 'Sub08.nii', 'Sub10.nii', 'Sub11.nii', 'Sub12.nii', 'Sub14.nii', 'Sub15.nii', 'Sub16.nii', 'Sub17.nii', 'Sub18.nii', 'Sub19.nii', 'Sub20.nii', 'Sub21.nii', 'Sub22.nii', 'Sub23.nii', 'Sub24.nii', 'Sub25.nii', 'Sub26.nii', 'Sub27.nii', 'Sub28.nii', 'Sub29.nii', 'Sub30.nii', 'Sub31.nii', 'Sub32.nii', 'Sub33.nii', 'Sub34.nii', 'Sub35.nii', 'Sub36.nii', 'Sub37.nii', 'Sub38.nii', 'Sub40.nii', 'Sub42.nii', 'Sub44.nii', 'Sub45.nii', 'Sub46.nii', 'Sub47.nii', 'Sub48.nii', 'Sub49.nii', 'Sub50.nii', 'Sub51.nii', 'Sub53.nii', 'Sub54.nii', 'Sub55.nii', 'Sub56.nii', 'Sub57.nii', 'Sub58.nii', 'Sub59.nii', 'Sub60.nii', 'Sub61.nii', 'Sub62.nii', 'Sub63.nii', 'Sub66.nii', 'Sub67.nii', 'Sub68.nii', 'Sub69.nii', 'Sub70.nii', 'Sub71.nii', 'Sub72.nii', 'Sub73.nii', 'Sub74.nii', 'Sub75.nii', 'Sub76.nii', 'Sub77.nii', 'Sub78.nii', 'Sub79.nii', 'Sub80.nii', 'Sub81.nii', 'Sub82.nii', 'Sub83.nii', 'Sub84.nii', 'Sub85.nii', 'Sub87.nii', 'Sub88.nii', 'Sub89.nii', 'Sub90.nii', 'Sub91.nii', 'Sub92.nii', 'Sub93.nii', 'Sub94.nii', 'Sub95.nii', 'Sub98.nii', 'Sub99.nii', 'Sub100.nii', 'Sub103.nii', 'Sub105.nii', 'Sub106.nii', 'Sub107.nii', 'Sub108.nii', 'Sub109.nii', 'Sub110.nii', 'Sub111.nii', 'Sub112.nii', 'Sub113.nii']
    dataset_list = ['Sub200.nii', 'Sub201.nii', 'Sub202.nii', 'Sub203.nii', 'Sub204.nii', 'Sub205.nii', 'Sub206.nii', 'Sub208.nii', 'Sub209.nii', 'Sub210.nii', 'Sub211.nii', 'Sub214.nii', 'Sub215.nii', 'Sub217.nii', 'Sub223.nii', 'Sub225.nii', 'Sub226.nii', 'Sub227.nii', 'Sub228.nii', 'Sub230.nii', 'Sub231.nii', 'Sub232.nii', 'Sub234.nii', 'Sub235.nii', 'Sub236.nii', 'Sub238.nii', 'Sub239.nii', 'Sub240.nii', 'Sub241.nii', 'Sub242.nii', 'Sub244.nii', 'Sub245.nii', 'Sub246.nii', 'Sub247.nii', 'Sub248.nii', 'Sub249.nii', 'Sub250.nii', 'Sub251.nii', 'Sub252.nii', 'Sub253.nii', 'Sub254.nii', 'Sub255.nii', 'Sub256.nii', 'Sub257.nii', 'Sub258.nii', 'Sub259.nii', 'Sub260.nii', 'Sub262.nii', 'Sub263.nii', 'Sub264.nii']
    
    for lst in dataset_list:
        # path_to_label = f'./Dataset/BULLEYE_Unet3_mask_new/{lst}'
        # path_to_prediction = f'./Dataset/BULLEYE_Unet2_mask_new/{lst}'
        # path_to_prediction = f'./Dataset/BULLEYE_mask/{lst}'

        path_to_label = f'./Dataset/ALMAZ_mask/{lst}'
        path_to_prediction = f'./Dataset/BULLEYE_mask/{lst}'

        # path_to_label = f'./Dataset/HCM_adult_mask/{lst}'
        # path_to_prediction = f'./Dataset/HCM_adult_Unet5_mask_new/{lst}'

        cm = CountRelVolume(path_to_label, path_to_prediction)
        cm.print()

