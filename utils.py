import numpy as np
import pickle, os

def slice_normalization(img):
    slice_num = img.shape[-1]
    for i in range(slice_num):
        target_img = img[..., i]
        img[..., i] = np.float32((target_img - np.min(target_img)) / (np.max(target_img) - np.min(target_img)))
    return img


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