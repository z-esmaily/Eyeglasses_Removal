# Enhancing Eyeglasses Removal in Facial Images Using Eyeglasses Mask Completion
Official pytorch implementation of paper "Enhancing Eyeglasses Removal in Facial Images: A Novel Approach Using Translation Models for Eyeglasses Mask Completion". (  2024)

## Quick Usage
Our Eyeglasses Removal method enhances this work: [Portrait Eyeglasses and Shadow Removal by Leveraging 3D Synthetic Data](https://github.com/StoryMY/take-off-eyeglasses)
They sharied their glass removal pretrained model in [here](https://drive.google.com/file/d/1Ea8Swdajz2J5VOkaXIw_-pVJk9EWYrpx/view?usp=sharing) Download it and put it in the "take-off-eyeglasses/ckpt" directory.
We add "Mask Completion" and "Post-Process" steps to it that improved quantitative metrics (FID, KID) and qualitative evaluations. Download Mask Compleetion model from [here](https://drive.google.com/file/d/1U-hanxKcG-chfUzxQV3G_Q7IBbNlHga3/view?usp=sharing), and put it in the "./PIX2PIX/log" directory. 

By default, Mask completion and Post-Process steps are active. If you want deactive them, you can set to False this arguments relevantly: --completion , --post_process in simple_take-off-eyeglasses.ipynb code
Our default input and result directories are:   ./TestDataAndResults/with_glasses  and  ./TestDataAndResults/removed_by_prop respectivly. 
But you can change them by --input_dir and --save_dir arguments as you want.
for simple usage put your input data in "with_glasses" folder and run:    simple_take-off-eyeglasses.ipynb
But you can also change different arguments in this command of run code:    !python3 easy_use_proposed.py --input_dir (your input path) --save_dir (your result path) --completion (by default is True) --post_process (by default is True)


## Test Only Mask completion (without eyeglasses removal)
 Download the pretrained Pix2Pix model from [here](https://drive.google.com/file/d/1U-hanxKcG-chfUzxQV3G_Q7IBbNlHga3/view?usp=sharing), and put it in the "./PIX2PIX/log" directory. Then run the: 
 TestPix2Pix_MaskCompletion.ipynb
 
## Paired Mask Dataset
Download the paired mask dataset in this [Google Drive](https://drive.google.com/drive/folders/1s3Vp-bpsMvo7DoY8f_yze_YBgMjeIZQI?usp=sharing).

## Citation

If our paper helps your research, please cite it in your publications:
	@inproceedings{lyu2022portrait,
	  title={Enhancing Eyeglasses Removal in Facial Images: A Novel Approach Using Translation Models for Eyeglasses Mask Completion},
	  author={Zahra Esmaeily, Hossein Ebrahimpour-Komleh},
	  booktitle={multimedia tools and applications},
	  pages={},
	  year={2024}
	}
