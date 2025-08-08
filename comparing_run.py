 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.4
Date: 04-05-2025
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""


import numpy as np

import CardioCascadeNet


class Statistics(CardioCascadeNet.MetaParameters):
	def __init__(self, subjects_list):
		super(CardioCascadeNet.MetaParameters, self).__init__()
		self.__subjects_list = subjects_list

	def get_stat_value(self, function):
		for clss in range(1, self.NUM_CLASS):
			summ_stat_value, summ_stat_value_pslc = [], []

			for sbjct in self.__subjects_list:	
				cm = CardioCascadeNet.CompareMatrix(f'{self.MASKS_DIR}/{sbjct}', f'{self.NEW_UNET3_MASKS_PATH}{sbjct}', clss)
				
				# cm = CardioCascadeNet.CompareMatrix(f'{self.MASKS_DIR}/{sbjct}', f'./Dataset/Almaz_masks_Max/{sbjct}', clss)
				# cm = CardioCascadeNet.CompareMatrix(f'./Dataset/Almaz_masks_Max/{sbjct}', f'{self.NEW_UNET3_MASKS_PATH}{sbjct}', clss)
				
				# cm = CardioCascadeNet.CompareBullsEyeMatrix(f'{self.DATASET_DIR}{self.DATASET_NAME}_mask_bullmasks/{sbjct}', f'{self.NEW_UNET5_MASKS_PATH}{sbjct}', clss)
				summ_stat_value.append(cm.stat_value(function))
				
				# print(cm)
				
				for stvl2d in cm.stat_value_2d(function):
					summ_stat_value_pslc.append(stvl2d)

			print(
				f'{function.upper()} Class_{self.DICT_CLASS[clss]}: Average per sub = {round(np.mean(summ_stat_value), 3)}, '
				f'Median per sub = {round(np.median(summ_stat_value), 3)}, \n'
				f'{function.upper()} Class_{self.DICT_CLASS[clss]}: Average per slice = {round(np.mean(summ_stat_value_pslc), 3)}, '
				f'Median per slice = {round(np.median(summ_stat_value_pslc), 3)} \n'
				)


class ComparingRun():
	def __init__(self):
		super().__init__()

	def comparing_run(self):
		jsnlst = CardioCascadeNet.JsonFoldList()
		
		test_list = jsnlst.load_dataset_list('test_list')
		subjects_list = test_list

		stats = Statistics(subjects_list)
		stats.get_stat_value('dice')
		stats.get_stat_value('precision')
		stats.get_stat_value('recall')



if __name__ == '__main__':	
	ComparingRun().comparing_run()


