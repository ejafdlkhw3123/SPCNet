from model.model import *
import numpy as np
import os
from skimage import io
from utils import *


class Tester(SPCNet):
    def __init__(self, cfg, device, dataset_test):
        super().__init__(device)
        self.device = device
        self.dataset_test = dataset_test
        self.init_params(cfg)
        self.load_epoch = cfg['load_epoch']
        self.path = cfg['load_path']
        self.size = cfg['size']
        self.vector_tensor = self.param_to_vector()

    def init_params(self, cfg):
        for key, value in cfg.items():
            setattr(self, key, value)

    def param_to_vector(self):
        # all case
        dict1 = {'Echo Time': 10.0, 'Repetition Time': 200.0, 'Inversion Time': 0.0, 'Flip Angle': 70.0, 'Magnetic Field Strength': 3.0}
        dict2 = {'Echo Time': 10.0, 'Repetition Time': 500.0, 'Inversion Time': 0.0, 'Flip Angle': 70.0,  'Magnetic Field Strength': 3.0}
        dict3 = {'Echo Time': 10.0, 'Repetition Time': 1000.0, 'Inversion Time': 0.0, 'Flip Angle': 70.0, 'Magnetic Field Strength': 3.0}
        dict4 = {'Echo Time': 70.0, 'Repetition Time': 3000.0, 'Inversion Time': 0.0, 'Flip Angle': 90.0,  'Magnetic Field Strength': 3.0}
        dict5 = {'Echo Time': 100.0, 'Repetition Time': 3000.0, 'Inversion Time': 0.0, 'Flip Angle': 90.0,  'Magnetic Field Strength': 3.0}
        dict6 = {'Echo Time': 130.0, 'Repetition Time': 3000.0, 'Inversion Time': 0.0, 'Flip Angle': 90.0,  'Magnetic Field Strength': 3.0}
        dict_list = [dict1, dict2, dict3, dict4, dict5, dict6]

        return self.dict_to_tensor(dict_list)

    def dict_to_tensor(self, dict_list):
        vector_list = []
        for dict in dict_list:
            vectors = np.array([dict['Repetition Time'], dict['Echo Time'] * 10, dict['Inversion Time'], dict['Flip Angle'], dict['Magnetic Field Strength'] * 10]) / 3000
            vector_list.append(vectors)
        vector_list = np.stack(vector_list, axis=0)
        return torch.tensor(vector_list).to(self.device).to(torch.float32)

    def load_model(self, load_epoch):
        self.encoder.load_state_dict(torch.load(self.load_path + '/weight/encoder_%d.pt' % load_epoch, map_location=self.device))
        self.generator.load_state_dict(torch.load(self.load_path + '/weight/generator_%d.pt' % load_epoch, map_location=self.device))

    def create_image(self):
        self.load_model(load_epoch=self.load_epoch)
        self.encoder.eval()
        self.generator.eval()

        with torch.no_grad():
            for idx, data in enumerate(self.dataset_test.load_test()):
                image = data.to(self.device)
                save_img = self.save_image(image)
                save_path = self.load_path + 'image/%d_%s/' % (self.load_epoch, self.save_path)
                create_emtpy_folder([save_path])
                io.imsave(save_path + '%03d.jpg' % idx, save_img)
                print(idx)

    def save_image(self, image):
        output_list, _, _ = self.middle_output(image)
        save_list = concat_images(image[0], output_list[0])
        save_img = np.float32(torch.concat([save_list], dim=0).cpu().detach().numpy())

        return np.uint8(save_img * 255)

    def middle_output(self, image):
        output_list, style_list, content_list = [], [], []
        for i in range(image.shape[0]):
            _image = image[[i],].repeat(self.vector_tensor.shape[0], 1, 1, 1)
            style_code, content_code, mid_layer = self.encoder(_image, self.vector_tensor)
            output = self.generator(style_code, content_code, mid_layer)
            output_list.append(output)
            style_list.append(style_code)
            content_list.append(content_code)
        return output_list, style_list, content_list




