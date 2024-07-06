import os
import cv2
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from pix2pix_model import Generator  # Import the Generator class

import argparse


def process_gen_image(result_image):
    kernel_closing = np.ones((3, 3), np.uint8)
    kernel_dilation = np.ones((10, 10), np.uint8)

    image = np.array(result_image)
    height, width, channels = image.shape

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_closing)
    img = cv2.cvtColor(closing, cv2.COLOR_GRAY2RGB)
    dilation = cv2.dilate(gray, kernel_dilation, iterations=1)
    ret, thresh = cv2.threshold(dilation, 100, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0 :
      x, y, w, h = 0, 0, width, height
    else:
      # Find the biggest contour (c) by the area
      cnt = max(contours, key=cv2.contourArea)
      x, y, w, h = cv2.boundingRect(cnt)

    blackImage = np.zeros_like(image)
    blackImage = cv2.rectangle(blackImage.copy(), (0, y), (width, y + h), (255, 255, 255), -1)
    newImage = cv2.bitwise_and(img, blackImage)

    return newImage


def mask_completion(completion_model_path, test_path, save_path, post_process):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load the model
    model = Generator().to(device)
    model.load_state_dict(torch.load(completion_model_path, map_location=torch.device('cpu')))
    model.eval()

    test_files = os.listdir(test_path)
    # Iterate over test images
    for file_name in test_files:
        if file_name.endswith('gmask.png'):  # Adjust the file extension as needed
            image_path = os.path.join(test_path, file_name)
            image = Image.open(image_path).convert('RGB')  # Open and convert the image to RGB

            image = test_transform(image)
            image = image.unsqueeze(0).to(device)
            # Perform inference with the model
            with torch.no_grad():
                generated_image = model(image)

            # Reverse the normalization for visualization
            generated_image = generated_image * 0.5 + 0.5

            # Convert the generated image to a NumPy array for visualization
            generated_image = generated_image.squeeze(0).cpu().numpy()
            generated_image = np.transpose(generated_image, (1, 2, 0))

            # Process glasses mask
            if post_process:
              completed_img = process_gen_image((generated_image * 255).astype(np.uint8))
            else:
              completed_img = (generated_image * 255).astype(np.uint8)

            save_img = os.path.join(save_path, file_name)
            cv2.imwrite(save_img, completed_img)

            # figs = plt.figure(figsize=(7, 7))
            # plt.subplot(1, 3, 1)
            # plt.axis("off")
            # plt.title("Original mask")
            # plt.imshow(np.transpose(vutils.make_grid(image, nrow=1, padding=2, normalize=True).cpu(), (1, 2, 0)))
            # plt.subplot(1, 3, 2)
            # plt.axis("off")
            # plt.title("Completed mask")
            # plt.imshow(generated_image)
            # plt.subplot(1, 3, 3)
            # plt.axis("off")
            # plt.title("After postprocess")
            # plt.imshow(proc_gen_img)
            # plt.show()
            # plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, default="./input_masks", help="input dir")
    parser.add_argument("--completion_model_path", type=str, default="./log/pix2pix_epoch17.pth", help="checkpoint of the model")
    parser.add_argument("--post_process", type=bool, default=True, help="if ture do post process")
    parser.add_argument("--save_path", type=str, default="./completed_masks", help="result dir")   
    args = parser.parse_args()
    print("Call with args:")
    print(args)

    mask_completion(args.completion_model_path, args.test_path, args.save_path, args.post_process)
