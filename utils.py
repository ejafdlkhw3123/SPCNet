import numpy as np
import os, torch

def create_emtpy_folder(path_list):
    for path in path_list:
        if not os.path.exists(path):
            os.makedirs(path)


def img_normal(img):
    img[img < 0] = 0
    img[img > 1] = 1
    return (img - torch.min(img)) / (torch.max(img) - torch.min(img))


def concat_images(image, output_images):
    save_img = image.squeeze()
    for i in range(output_images.shape[0]):
        output = img_normal(output_images[i].squeeze())
        save_img = torch.concat([save_img, output], dim=1)
    return save_img
