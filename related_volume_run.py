 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.4
Date: 04-05-2025
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""


import numpy as np
import nibabel as nib


import CardioCascadeNet



jsnlst = CardioCascadeNet.JsonFoldList()
dataset_list = jsnlst.load_dataset_list('test_list')
# dataset_list = jsnlst.load_dataset_list('train_list')
# dataset_list = jsnlst.load_dataset_list('valid_list')
jsnlst.pprint('test_list')
# jsnlst.pprint('train_list')
# jsnlst.pprint('valid_list')


class CountRelVolume(CardioCascadeNet.MetaParameters):
    def __init__(self, path_to_label: str, path_to_prediction: str):
        super(CardioCascadeNet.MetaParameters, self).__init__()

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
            dictionary[f'rVlm_{self.DICT_CLASS[key]}'] = 0

        for slc in range(self.length):
            fibrosis = self.fibrosis[:,:,:]
            bulleye = self.bulleye[:,:,:]
            
        for key in range(1, num_class):
            fibrosis_ = fibrosis.copy()
            bulleye_ = bulleye.copy()

            bulleye_[bulleye_ != key] = 0
            bulleye_[bulleye_ == key] = 1
            fibrosis_ = fibrosis_ * bulleye_

            fib = fibrosis_[fibrosis_ == 3]
            myo = fibrosis_[fibrosis_ == 2]

            fib[fib!=3] = 0
            fib[fib!=0] = 1

            myo[myo!=2] = 0
            myo[myo!=0] = 1

            fib = fib.sum().item()
            myo = myo.sum().item()

            rel_volume = int((fib) / (fib + myo + self.smooth) * 100)
            dictionary[f'rVlm_{self.DICT_CLASS[key]}'] = rel_volume
        
        return dictionary

    def rel_aortic_volume(self):
        num_class = len(self.DICT_CLASS)
        dictionary = {}

        for key in range(1, num_class):
            dictionary[f'rVlm_{self.DICT_CLASS[key]}'] = 0

        for slc in range(self.length):
            fibrosis = self.fibrosis[:,:,:]
            bulleye = self.bulleye[:,:,:]

        bulleye[bulleye == 1] = 1
        bulleye[bulleye == 2] = 1
        bulleye[bulleye == 3] = 2
        bulleye[bulleye == 4] = 2
        bulleye[bulleye == 5] = 3
        bulleye[bulleye == 6] = 3
        bulleye[bulleye == 7] = 1
        bulleye[bulleye == 8] = 1
        bulleye[bulleye == 9] = 2
        bulleye[bulleye == 10] = 2
        bulleye[bulleye == 11] = 3
        bulleye[bulleye == 12] = 3
        bulleye[bulleye == 13] = 1
        bulleye[bulleye == 14] = 1
        bulleye[bulleye == 15] = 2
        bulleye[bulleye == 16] = 3
        bulleye[bulleye == 17] = 4

        # bulleye[bulleye != 1] = 1

        for key in range(1, num_class):
            fibrosis_ = fibrosis.copy()
            bulleye_ = bulleye.copy()

            bulleye_[bulleye_ != key] = 0
            bulleye_[bulleye_ == key] = 1

            fibrosis_ = fibrosis_ * bulleye_

            fib = fibrosis_[fibrosis_ == 3]
            myo = fibrosis_[fibrosis_ == 2]

            fib[fib!=3] = 0
            fib[fib!=0] = 1

            myo[myo!=2] = 0
            myo[myo!=0] = 1

            fib = fib.sum().item()
            myo = myo.sum().item()

            rel_volume = int((fib) / (fib + myo + self.smooth) * 100)
            dictionary[f'rVlm_{self.DICT_CLASS[key]}'] = rel_volume
        
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

                seg_check = CardioCascadeNet.InstancesFinder(fibrosis_new, kernel = 144, num_class = 3).transcheck()
                dictionary[f'{self.DICT_CLASS[key]}'] += ([i for i in seg_check])
        
        for key in range(1, num_class):
            dictionary[f'{self.DICT_CLASS[key]}'] = list(set(dictionary[f'{self.DICT_CLASS[key]}']))
            if len(dictionary[f'{self.DICT_CLASS[key]}']) > 1:
                dictionary[f'{self.DICT_CLASS[key]}'].remove('-')

        return dictionary

    def print(self):
        print(self.sub_name())
        print(self.rel_volume())
        # print(self.rel_aortic_volume())
        # print(self.check_transmural())
        print()


class CountRelvolumeRun():
    def __init__(self):
        super().__init__()

    def count_rel_volume_run(self):
        # dataset_list = ['Sub01.nii', 'Sub02.nii', 'Sub03.nii', 'Sub04.nii', 'Sub05.nii', 'Sub06.nii', 'Sub07.nii', 'Sub08.nii', 'Sub10.nii', 'Sub11.nii', 'Sub12.nii', 'Sub14.nii', 'Sub15.nii', 'Sub16.nii', 'Sub17.nii', 'Sub18.nii', 'Sub19.nii', 'Sub20.nii', 'Sub21.nii', 'Sub22.nii', 'Sub23.nii', 'Sub24.nii', 'Sub25.nii', 'Sub26.nii', 'Sub27.nii', 'Sub28.nii', 'Sub29.nii', 'Sub30.nii', 'Sub31.nii', 'Sub32.nii', 'Sub33.nii', 'Sub34.nii', 'Sub35.nii', 'Sub36.nii', 'Sub37.nii', 'Sub38.nii', 'Sub40.nii', 'Sub42.nii', 'Sub44.nii', 'Sub45.nii', 'Sub46.nii', 'Sub47.nii', 'Sub48.nii', 'Sub49.nii', 'Sub50.nii', 'Sub51.nii', 'Sub53.nii', 'Sub54.nii', 'Sub55.nii', 'Sub56.nii', 'Sub57.nii', 'Sub58.nii', 'Sub59.nii', 'Sub60.nii', 'Sub61.nii', 'Sub62.nii', 'Sub63.nii', 'Sub66.nii', 'Sub67.nii', 'Sub68.nii', 'Sub69.nii', 'Sub70.nii', 'Sub71.nii', 'Sub72.nii', 'Sub73.nii', 'Sub74.nii', 'Sub75.nii', 'Sub76.nii', 'Sub77.nii', 'Sub78.nii', 'Sub79.nii', 'Sub80.nii', 'Sub81.nii', 'Sub82.nii', 'Sub83.nii', 'Sub84.nii', 'Sub85.nii', 'Sub87.nii', 'Sub88.nii', 'Sub89.nii', 'Sub90.nii', 'Sub91.nii', 'Sub92.nii', 'Sub93.nii', 'Sub94.nii', 'Sub95.nii', 'Sub98.nii', 'Sub99.nii', 'Sub100.nii', 'Sub103.nii', 'Sub105.nii', 'Sub106.nii', 'Sub107.nii', 'Sub108.nii', 'Sub109.nii', 'Sub110.nii', 'Sub111.nii', 'Sub112.nii', 'Sub113.nii']
        # dataset_list = ['Sub200.nii', 'Sub201.nii', 'Sub202.nii', 'Sub203.nii', 'Sub204.nii', 'Sub205.nii', 'Sub206.nii', 'Sub208.nii', 'Sub209.nii', 'Sub210.nii', 'Sub211.nii', 'Sub214.nii', 'Sub215.nii', 'Sub217.nii', 'Sub223.nii', 'Sub225.nii', 'Sub226.nii', 'Sub227.nii', 'Sub228.nii', 'Sub230.nii', 'Sub231.nii', 'Sub232.nii', 'Sub234.nii', 'Sub235.nii', 'Sub236.nii', 'Sub238.nii', 'Sub239.nii', 'Sub240.nii', 'Sub241.nii', 'Sub242.nii', 'Sub244.nii', 'Sub245.nii', 'Sub246.nii', 'Sub247.nii', 'Sub248.nii', 'Sub249.nii', 'Sub250.nii', 'Sub251.nii', 'Sub252.nii', 'Sub253.nii', 'Sub254.nii', 'Sub255.nii', 'Sub256.nii', 'Sub257.nii', 'Sub258.nii', 'Sub259.nii', 'Sub260.nii', 'Sub262.nii', 'Sub263.nii', 'Sub264.nii']
        # dataset_list = ['Sub35.nii', 'Sub07.nii', 'Sub112.nii', 'Sub99.nii', 'Sub76.nii', 'Sub56.nii', 'Sub111.nii', 'Sub85.nii', 'Sub66.nii', 'Sub32.nii', 'Sub53.nii', 'Sub83.nii', 'Sub61.nii', 'Sub49.nii', 'Sub42.nii', 'Sub14.nii', 'Sub69.nii', 'Sub105.nii', 'Sub03.nii', 'Sub23.nii']
        # dataset_list = ['AMOSOVA_V_A_056__tfi2d1_76.nii', 'EFIMOVA_E_N__011__tfi2d1_70.nii', 'GAVRILIN_A_V__030__tfi2d1_70.nii', 'MARKOVA_E_A__030__tfi2d1_70.nii', 'SAYADOV_K_M__053__tfi2d1_82.nii', 'SLAVUTSKII_V_A__019__tfi2d1_70.nii', 'SOTIN_V_I__030__tfi2d1_70.nii', 'TEIMUROV_G_S_O_036__tfi2d1_70.nii']
        # dataset_list = ['Sub03.nii']
        dataset_list = ['SubHCM001.nii', 'SubHCM002.nii', 'SubHCM003.nii', 'SubHCM004.nii', 'SubHCM005.nii', 'SubHCM006.nii', 'SubHCM007.nii', 'SubHCM008.nii', 'SubHCM009.nii', 'SubHCM010.nii', 'SubHCM011.nii', 'SubHCM012.nii', 'SubHCM013.nii', 'SubHCM014.nii', 'SubHCM015.nii', 'SubHCM016.nii', 'SubHCM017.nii', 'SubHCM018.nii', 'SubHCM019.nii', 'SubHCM020.nii', 'SubHCM021.nii', 'SubHCM022.nii', 'SubHCM023.nii', 'SubHCM024.nii', 'SubHCM025.nii', 'SubHCM026.nii', 'SubHCM027.nii', 'SubHCM028.nii', 'SubHCM029.nii', 'SubHCM030.nii', 'SubHCM031.nii', 'SubHCM032.nii', 'SubHCM033.nii', 'SubHCM034.nii', 'SubHCM035.nii', 'SubHCM036.nii', 'SubHCM037.nii', 'SubHCM038.nii', 'SubHCM039.nii', 'SubHCM040.nii', 'SubHCM041.nii', 'SubHCM042.nii', 'SubHCM043.nii', 'SubHCM044.nii', 'SubHCM045.nii', 'SubHCM046.nii', 'SubHCM047.nii', 'SubHCM048.nii', 'SubHCM049.nii', 'SubHCM050.nii', 'SubHCM051.nii', 'SubHCM052.nii', 'SubHCM053.nii', 'SubHCM054.nii', 'SubHCM055.nii', 'SubHCM056.nii', 'SubHCM057.nii', 'SubHCM058.nii', 'SubHCM059.nii', 'SubHCM060.nii', 'SubHCM061.nii', 'SubHCM062.nii', 'SubHCM063.nii', 'SubHCM064.nii', 'SubHCM065.nii', 'SubHCM066.nii', 'SubHCM067.nii', 'SubHCM068.nii', 'SubHCM069.nii', 'SubHCM070.nii', 'SubHCM071.nii', 'SubHCM072.nii', 'SubHCM073.nii', 'SubHCM074.nii', 'SubHCM075.nii', 'SubHCM076.nii', 'SubHCM077.nii', 'SubHCM078.nii', 'SubHCM079.nii', 'SubHCM080.nii', 'SubHCM081.nii', 'SubHCM082.nii', 'SubHCM083.nii', 'SubHCM084.nii', 'SubHCM085.nii', 'SubHCM086.nii', 'SubHCM087.nii', 'SubHCM088.nii', 'SubHCM089.nii', 'SubHCM090.nii', 'SubHCM091.nii', 'SubHCM092.nii', 'SubHCM093.nii', 'SubHCM094.nii', 'SubHCM095.nii', 'SubHCM096.nii', 'SubHCM097.nii', 'SubHCM098.nii', 'SubHCM099.nii', 'SubHCM100.nii']
        
        for lst in dataset_list:
            # path_to_label = f'./Dataset/BULLEYE_Unet3_mask_new/{lst}'
            # path_to_prediction = f'./Dataset/BULLEYE_Unet2_mask_new/{lst}'
            # path_to_prediction = f'./Dataset/BULLEYE_mask/{lst}'

            # path_to_label = f'./Dataset/ALMAZ_Unet3_mask_new/{lst}'
            # path_to_prediction = f'./Dataset/ALMAZ_Unet5_mask_new/{lst}'

            # path_to_label = f'./Dataset/ALMAZ_mask/{lst}'
            # path_to_prediction = f'./Dataset/BULLEYE_mask/{lst}'

            path_to_label = f'./CardioCascadeNet/Dataset/HCM_adult_mask/{lst}'
            path_to_prediction = f'./CardioCascadeNet/Dataset/HCM_adult_mask_bullmasks/{lst}'

            cm = CountRelVolume(path_to_label, path_to_prediction)
            cm.print()


        

if __name__ == "__main__":
    CountRelvolumeRun().count_rel_volume_run()

