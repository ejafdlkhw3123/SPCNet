import numpy as np
import cv2, random, os, h5py, json
import torch.utils.data
import torch


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, input_path, size=256, slice_num=-1, shuffle=True):
        self.input_path = input_path
        self.size = size
        self.slice_num = slice_num
        self.train_list = self.select_name(input_path + '/train/', shuffle=shuffle)
        self.test_list = self.select_name(input_path + '/test/', shuffle=shuffle)
        self.total_num = len(self.train_list)

    def select_name(self, input_path, shuffle=True):
        input_list = os.listdir(input_path)
        input_list = [input_path + item for item in input_list]

        if shuffle:
            random.shuffle(input_list)
        if self.slice_num != -1:
            input_list = input_list[:self.slice_num]
        return input_list

    def load_sample(self):
        input_list, info_list = [], []
        index_list = random.sample(range(len(self.train_list)), 2)
        for index in index_list:
            input_data = h5py.File(self.train_list[index])
            input_img = self.img_resize(input_data['array'][:])
            input_img = torch.from_numpy(np.float32(input_img[np.newaxis, :, :]))
            input_list.append(input_img)
            info_list.append(str(json.loads(input_data.attrs['header'])))

        input_list = torch.stack(input_list, dim=0)

        return input_list, info_list

    def load_test(self):
        for name in self.test_list:
            input_data = h5py.File(name)
            input_img = self.img_resize(input_data['array'][:])
            input_img = torch.from_numpy(np.float32(input_img[np.newaxis, :, :]))

            yield input_img

    def img_resize(self, img):
        return self.img_nor(cv2.resize(img, (self.size, self.size)))

    def img_nor(self, img):
        return np.float32((img - np.min(img)) / (np.max(img) - np.min(img)))

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, index):
        input_data = h5py.File(self.train_list[index])

        input_img = np.float32(self.img_resize(input_data['array'][:]))
        input_img = input_img[np.newaxis, ]

        input_info = json.loads(input_data.attrs['header'])

        return input_img, str(input_info)

