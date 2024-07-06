import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from models.networks import ResnetGenerator, ResnetGeneratorMask
from models.domain_adaption import DomainAdapter
from torchvision import transforms
from PIL import Image

from eyeglass_mask_completion import mask_completion
# import time


def savetensor2img(tensor, save_path):
    out = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    out = (out + 1) / 2
    out = np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)
    if out.shape[-1] == 1:
        out = out[:, :, 0]
    out_PIL = Image.fromarray(out)
    out_PIL.save(save_path)

def tensor2img(tensor):
    out = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    out = (out + 1) / 2
    out = np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)
    if out.shape[-1] == 1:
        out = out[:, :, 0]
    out_PIL = Image.fromarray(out)
    return out_PIL

def out2mask(tensor, soft=True, scale=1.25):
    if soft:
        ret = (nn.Softmax2d()(tensor)[:, 1:2, :, :]) * scale
    else:
        ret = (tensor.argmax(1).unsqueeze(1).float())
    return ret

def smart_mkdir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def generate_results(args, device):
    norm = nn.BatchNorm2d
    DA_Net = DomainAdapter().to(device)
    GlassMask_Net = ResnetGeneratorMask(input_nc=64, output_nc=2, norm_layer=norm).to(device)  # shadow prediction (mask)
    ShadowMask_Net = ResnetGeneratorMask(input_nc=65, output_nc=2, norm_layer=norm).to(device)  # shadow prediction (mask)
    DeShadow_Net = ResnetGenerator(input_nc=5, output_nc=3, norm_layer=norm).to(device)
    DeGlass_Net = ResnetGenerator(input_nc=4, output_nc=3, norm_layer=norm).to(device)

    # load ckpt
    if device == "cuda": 
      ckpt = torch.load(args.ckpt_path) 
    else:
      ckpt = torch.load(args.ckpt_path, map_location=torch.device('cpu'))

    DA_Net.load_state_dict(ckpt["DA"])
    DA_Net.eval()
    GlassMask_Net.load_state_dict(ckpt["GM"])
    GlassMask_Net.eval()
    ShadowMask_Net.load_state_dict(ckpt["SM"])
    ShadowMask_Net.eval()
    DeShadow_Net.load_state_dict(ckpt["DeShadow"])
    DeShadow_Net.eval()
    DeGlass_Net.load_state_dict(ckpt["DeGlass"])
    DeGlass_Net.eval()

    # transform
    transform = transforms.Compose([
        transforms.Resize([args.img_size, args.img_size]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    transform2 = transforms.Compose([
        transforms.ToTensor()
    ])

    img_list = os.listdir(args.input_dir)
    with torch.no_grad():
        for img_name in img_list:
            img = Image.open(os.path.join(args.input_dir, img_name)).convert('RGB')
            img = transform(img)
            img = torch.unsqueeze(img, 0)
            img = img.to(device)

            gfm, sfm = DA_Net(img)
            gmask = out2mask(GlassMask_Net(gfm), False)
            
            if (args.completion == False and args.post_process == False):
              gmaskpro = gmask
            else: 
              gmaskpro_array = mask_completion(gmask * 2 - 1, completion=args.completion, completion_model_path = args.completion_model_dir, post_process = args.post_process)
            gmaskpro = Image.fromarray(gmaskpro_array).convert('L')

            gmaskpro = transform2(gmaskpro)
            gmaskpro = torch.unsqueeze(gmaskpro, 0)
            gmaskpro = gmaskpro.to(device)

            smask = out2mask(ShadowMask_Net(torch.cat([sfm, gmaskpro], dim=1)), True)

            ds_in = torch.cat([img, smask, gmaskpro], dim=1)
            ds_out = DeShadow_Net(ds_in)
            ds_out_masked = ds_out * (1 - gmaskpro)
            dg_in = torch.cat([ds_out_masked, gmaskpro], dim=1)
            dg_out = DeGlass_Net(dg_in)

            savetensor2img(dg_out, os.path.join(args.save_dir, img_name))
            savetensor2img(gmaskpro * 2 - 1, os.path.join(args.save_dir, img_name[:-4] + '_gmask.png'))
            savetensor2img(smask * 2 - 1, os.path.join(args.save_dir, img_name[:-4] + '_smask.png'))

if __name__ == '__main__':
    device = "cpu"
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, default="../TestDataAndResults/CelebA/with_glasses", help="input dir")
    parser.add_argument("--completion_model_dir", type=str, default="../PIX2PIX/log/pix2pix_epoch17.pth", help="mask completion model dir")
    parser.add_argument("--completion", type=bool, default=True, help="if ture do completion")
    parser.add_argument("--post_process", type=bool, default=True, help="if ture do post process")
    parser.add_argument("--save_dir", type=str, default="../TestDataAndResults/CelebA/removed_by_prop", help="result dir")
    
    parser.add_argument("--img_size", type=int, default=256, help="image sizes for the model")
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/pretrained.pt", help="checkpoint of the model")
    args = parser.parse_args()
    print("Call with args:")
    print(args)

    smart_mkdir(args.save_dir)
    generate_results(args, device)
