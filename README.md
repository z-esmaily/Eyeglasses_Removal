# Enhancing Eyeglasses Removal in Facial Images Using Eyeglasses Mask Completion
Official pytorch implementation of paper "Enhancing Eyeglasses Removal in Facial Images: A Novel Approach Using Translation Models for Eyeglasses Mask Completion". (  2024)

## Quick Usage
Our Eyeglasses Removal method enhances this work:"[Portrait Eyeglasses and Shadow Removal by Leveraging 3D Synthetic Data]" https://github.com/StoryMY/take-off-eyeglasses
They sharied their glass removal pretrained model in [here](https://drive.google.com/file/d/1Ea8Swdajz2J5VOkaXIw_-pVJk9EWYrpx/view?usp=sharing) Download it and put it in the "take-off-eyeglasses/ckpt" directory.

We add "Mask Completion" and "Post-Process" steps to it that improved quantitative metrics (FID, KID) and qualitative evaluations. Download Mask Compleetion model from [here](https://drive.google.com/file/d/1U-hanxKcG-chfUzxQV3G_Q7IBbNlHga3/view?usp=sharing), and put it in the "./PIX2PIX/log" directory. Then run: 
simple_take-off-eyeglasses.ipynb

By default, Mask completion and Post-Process steps are active. If you want deactive them, you can set to False this arguments relevantly: --completion , --post_process in this command: !python3 easy_use_proposed.py

As default, the input images are expected to be put in "data" directory and the results will be saved in "results" directory. You can also change them by different arguments.

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
