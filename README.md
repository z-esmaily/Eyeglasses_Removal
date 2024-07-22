# Enhancing Eyeglasses Removal in Facial Images Using Eyeglasses Mask Completion

This work presents an enhanced approach for eyeglasses removal in facial images. It builds upon existing methods by incorporating novel **mask completion and post-processing steps**, leading to improved quality and quantitative metrics.

**Official PyTorch implementation of the paper "Enhancing Eyeglasses Removal in Facial Images: A Novel Approach Using Translation Models for Eyeglasses Mask Completion" (Multimedia Tools and Applications, under review, 2024).**

## Method and Arguments
Our approach leverages the existing eyeglasses removal method in [*Portrait Eyeglasses and Shadow Removal by Leveraging 3D Synthetic Data*](https://github.com/StoryMY/take-off-eyeglasses)  (baseline method) as a foundation. However, we introduce a novel combination of mask completion and post-processing steps to achieve significantly improved quantitative metrics (FID, KID) and qualitative evaluations.

Below is some examples for mask completion and post-processing:
![Eyeglasses Removal Using Mask Completion_for github](https://github.com/user-attachments/assets/829eff83-39fb-4591-9619-06246bf42a7d)
![Post-processing of eyeglasses mask images_for github](https://github.com/user-attachments/assets/a738923c-7ff6-462b-aa94-42aebf86f7a3)

## Quick Usage
### 1. Prerequisites:

Download the pre-trained model from the baseline method [google drive link](https://drive.google.com/file/d/1Ea8Swdajz2J5VOkaXIw_-pVJk9EWYrpx/view?usp=sharing) and place it in the `"take-off-eyeglasses/ckpt"` directory.
Download the pre-trained Pix2Pix model we provide for mask completion from our [google drive link](https://drive.google.com/file/d/1U-hanxKcG-chfUzxQV3G_Q7IBbNlHga3/view?usp=sharing) and place it in the `"PIX2PIX/log"` directory.

### 2. Running the Code:

Place your input images with eyeglasses in the `"TestDataAndResults/with_glasses"` folder. Navigate to the `"take-off-eyeglasses"` folder and run the following notebook:

	simple_take-off-eyeglasses.ipynb.

Building upon the initial eyeglasses removal achieved by the pre-trained model from the baseline method, this notebook incorporates our mask completion and post-processing steps for enhanced results.

### 3. Optional Arguments (in simple_take-off-eyeglasses.ipynb):

By default, both mask completion and post-processing are active. You can adjust these functionalities using the following arguments:

--completion: Set to False to disable mask completion.

--post_process: Set to False to disable post-processing.

In other words, you can change arguments in the notebook using:

	!python3 easy_use_proposed.py --input_dir (your input path) --save_dir (your result path) --completion (by default is True) --post_process (by default is True)

## Test Only Mask completion (without eyeglasses removal)
Download the pre-trained Pix2Pix model from [here](https://drive.google.com/file/d/1U-hanxKcG-chfUzxQV3G_Q7IBbNlHga3/view?usp=sharing), and place it in the `"PIX2PIX/log"` directory.
 
Then run:    
 	
	TestPix2Pix_MaskCompletion.ipynb
 
## Paired Mask Dataset
![Examples of eyeglasses masks](https://github.com/user-attachments/assets/c2834dbd-9a7b-40a8-a1a0-6e7e5eef4cd9)

Download the paired mask dataset from [here](https://drive.google.com/drive/folders/1s3Vp-bpsMvo7DoY8f_yze_YBgMjeIZQI?usp=sharing).

To create your own paired mask dataset using Top-Hat morphological operation, run:   

	create_pair_samples_masks.ipynb

## Citation

If our paper helps your research, please cite it in your publications:

	@article{esmaeily2024enhancing_eyeglasses_removal,
	  title={Enhancing Eyeglasses Removal in Facial Images: A Novel Approach Using Translation Models for Eyeglasses Mask Completion},
	  author={Zahra Esmaeily and Hossein Ebrahimpour-Komleh},
	  journal={Multimedia Tools and Applications},
	  year={2024},
	  note={under review}
	}


