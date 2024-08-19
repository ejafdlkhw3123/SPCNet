import numpy as np
import pickle, os, torch


def load_pickle(folder_name, file_name):
    # file name load
    with open('%s/train_%s.pkl' % (folder_name, file_name), 'rb') as f:
        train_list = pickle.load(f)
        f.close()
    with open('%s/test_%s.pkl' % (folder_name, file_name), 'rb') as f:
        test_list = pickle.load(f)
        f.close()
    return train_list, test_list


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
