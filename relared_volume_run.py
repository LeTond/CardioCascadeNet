from Validation.comparing import *
from Preprocessing.split_dataset import *


jsnlst = JsonFoldList()
dataset_list = jsnlst.load_dataset_list('test_list')
jsnlst.pprint('test_list')


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
            dictionary[f'RelVol_{self.DICT_CLASS[key]}'] = round(float(rel_volume), 1)
        
        return dictionary

    def print(self):
        print(self.sub_name())
        print(self.rel_volume())
        print()


if __name__ == "__main__":
    for lst in dataset_list:
        # path_to_label = f'./Dataset/BULLEYE_Unet3_mask_new/{lst}'
        # path_to_prediction = f'./Dataset/BULLEYE_Unet2_mask_new/{lst}'
        # path_to_prediction = f'./Dataset/BULLEYE_mask/{lst}'

        path_to_label = f'./Dataset/HCM_adult_mask/{lst}'
        path_to_prediction = f'./Dataset/HCM_adult_Unet5_mask_new/{lst}'

        cm = CountRelVolume(path_to_label, path_to_prediction)
        cm.print()



# 