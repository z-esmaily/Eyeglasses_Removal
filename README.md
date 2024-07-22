# Enhancing Eyeglasses Removal in Facial Images Using Eyeglasses Mask Completion

**Abstract:**  
This work presents an enhanced approach for eyeglasses removal in facial images. It builds upon existing methods by incorporating eyeglasses mask completion and post-processing steps, leading to improved quality and quantitative metrics.

**Official PyTorch implementation of the paper "Enhancing Eyeglasses Removal in Facial Images: A Novel Approach Using Translation Models for Eyeglasses Mask Completion" (Multimedia Tools and Applications, under review, 2024).**

## Method and Arguments
Our Eyeglasses Removal method enhances this work: [*Portrait Eyeglasses and Shadow Removal by Leveraging 3D Synthetic Data*](https://github.com/StoryMY/take-off-eyeglasses), which removes eyeglasses using their trained model. 
Download this pre-trained model [here](https://drive.google.com/file/d/1Ea8Swdajz2J5VOkaXIw_-pVJk9EWYrpx/view?usp=sharing) and place it in the `"take-off-eyeglasses/ckpt"` directory.

We add "Mask Completion" and "Post-Process" steps that significantly improve quantitative metrics (FID, KID) and qualitative evaluations.

Download the Mask Completion model from [here](https://drive.google.com/file/d/1U-hanxKcG-chfUzxQV3G_Q7IBbNlHga3/view?usp=sharing), and place it in the `"PIX2PIX/log"` directory.

Our default input and result directories are:

- `"TestDataAndResults/with_glasses"` (input)
- `"TestDataAndResults/removed_by_prop"` (results)

These can be changed using the `--input_dir` and `--save_dir` arguments.

Below is some examples for mask completion and post-processing:
![Eyeglasses Removal Using Mask Completion_for github](https://github.com/user-attachments/assets/829eff83-39fb-4591-9619-06246bf42a7d)
![Post-processing of eyeglasses mask images_for github](https://github.com/user-attachments/assets/a738923c-7ff6-462b-aa94-42aebf86f7a3)

## Quick Usage
Place your input data in the `"TestDataAndResults/with_glasses"` folder. Run following notebook in '"take-off-eyeglasses" folder 
(take off eyeglasses codes in this folder based on [this](https://github.com/StoryMY/take-off-eyeglasses) repo.):

	simple_take-off-eyeglasses.ipynb.

Note: By default, Mask completion and Post-Process steps are active. To deactivate them, set the completion and post_process arguments to False in the simple_take-off-eyeglasses.ipynb code.

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


