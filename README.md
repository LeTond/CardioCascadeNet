# CardioCascadeUnet v1.6: 

### The CardioCascadeUnet framework is designed for training and inferencing of the cascade architecture based on the AttentionU-Net model for the tasks of segmenting and localizing myocardial scar tissue in LGE (late gadolinium enhanced) magnetic resonance images of the heart

![Cascade-logo_01](images/Picture_bull'e_eye.png)

*******************************************************************************
## Downloading models weights:
From CascadeCardioUnet root directory 
(wget is required)

In MacOs:
```Shell
brew install wget
```
or in Linux
```Shell
sudo apt install wget
```

Further:
```Shell
cd ./Model
chmod +x get_weights.sh
./get_weights.sh
cd .. 
```

*******************************************************************************
# Installation:
## Conda Env creating and installing requirements
```Shell
conda create -n cardio python=3.9
conda activate cardio
```

## Requirements
```Shell
pip install -r requirements.txt
```

Also training and inference of networks was tested on macOS Sonoma, M1 cpu with 
pytorch with mps technology support

macOS-14.6.1-arm64-arm-64bit
python version: 3.9.16 (main, Mar  8 2023, 04:29:24) 
torch version: 2.0.0
numpy version: 1.24.2

*******************************************************************************
# Neural Networks
## UNET1
- In MetaParameters class use UNET1 = True for training or prediction 
with default KERNEL (192x192 pixels). It is used to find the center of mass of 
the heart.

## UNET2
- The second model uses a cropped matrix of 64x64 pixels to improve scar 
and myocardium tissue segmentation.

## UNET3
- The third model uses a cropped matrix of 64x64 pixels with the removal 
of information from the image outside the gap (default is 8 pixels), to improve 
scar and myocardial tissue segmentation in the apical slices.

## UNET4
- It used to identify the level of the LV in each slice (basal, 
medial, apical, apex). 

## UNET5
- The last model, which divided the myocardium into 17 segments. 


*******************************************************************************
## Functions of Python scripts

- configurarion.py:	 Main list of options used to train the network.

- trainig_run.py:	Runs train of network. And it can be used to switch and 
choose a model configuration.

- validation_run.py:	

- inference_run.py:	It launches the inference of all images in the 
./Dataset/PROJECT_NAME_origin_new directory.

- comparing_run.py:	It launches a comparison of the predicted and reference
masks.

*******************************************************************************
*******************************************************************************
## Run modules

From root project directory run necessary modules:
	
 	.
	├── MainProject
	|   ├── main.py
	|   ├── CardioCascadeNet

main.py:

 	import CardioCascadeNet
 
	if __name__ == "__main__":
	    CardioCascadeNet.TrainRun().rewrite_weights_run()
	    CardioCascadeNet.TrainRun().train_run()
	    CardioCascadeNet.InferenceRun().run_process()
	    CardioCascadeNet.ComparingRun().comparing_run()
	    CardioCascadeNet.CountRelvolumeRun().count_rel_volume_run()
	    CardioCascadeNet.ValidationRun().validation_run()

## Run training and inference: 
```Shell
python main.py
```


*******************************************************************************
### Directories structure:

Data directories should have a similar structure:

	├── MainProject
 	|   ├── main.py
	|   ├── CardioCascadeNet
	|   ├── Dataset
	|   |   ├── PROJ_NAME_images
	|   |   |   ├── subname_01.nii.gz
	|   |   |   └── subname_02.nii.gz
	|   |   |   ├── subname_03.nii.gz
	|   |   |   └── subname_04.nii.gz
	|	|   |
	|   |   ├── PROJ_NAME_masks
	|   |   |   ├── subname_01.nii.gz
	|   |   |   └── subname_02.nii.gz
	|   |   |   ├── subname_03.nii.gz
	|   |   |   └── subname_04.nii.gz
	|	|   |
	|	|   ├── PROJ_NAME_masks_bullmasks
	|   |   |   ├── subname_01.nii.gz
	|   |   |   └── subname_02.nii.gz
	|   |   |   ├── subname_03.nii.gz
	|   |   |   └── subname_04.nii.gz
	|	|   |
	|   |   ├── PROJ_NAME_images_new
	|   |   |   ├── new_subname_01.nii.gz
	|   |   |   └── new_subname_02.nii.gz
	|   |   |   ├── new_subname_03.nii.gz
	|   |   |   └── new_subname_04.nii.gz
	|	|   |
	|   ├── Results
	|   |   ├── PROJ_NAME_Unet1_masks_new
	|   |   ├── PROJ_NAME_Unet2_masks_new
	|   |   ├── PROJ_NAME_Unet3_masks_new
	|   |   ├── PROJ_NAME_Unet4_masks_new
	|   |   ├── PROJ_NAME_Unet5_masks_new

 
### Training:

	CardioCascadeNet.TrainRun().train_run()



### Inference:
 - You can launch inference_run.py for testing cascade network on a new image



## References:
* https://doi.org/10.1016/j.bspc.2025.107555
* https://doi.org/10.17586/0021-3454-2025-68-11-996-1005
* https://www.researchgate.net/publication/380945819_Automatic_and_semi-automatic_segmentation_method_of_post-myocardial_infarction_according_to_magnetic_resonance_imaging_with_late_gadolinium_enhancement

