import time, io, re, torch
from model.model import *
from utils import create_emtpy_folder
import matplotlib.pyplot as plt
from open_clip import create_model_from_pretrained, get_tokenizer
import numpy as np
from skimage import io
from torch.utils.data import DataLoader
import pandas as pd
from openpyxl import load_workbook


class Train(MrStyler):
    def __init__(self, cfg, device, dataset_train):
        super().__init__(device)
        self.device = device
        self.dataset_train = dataset_train
        self.init_params(cfg)
        self.model_names = ['encoder', 'generator', 'param_extractor', 'discriminator1', 'discriminator2']
        self.model_list = [self.encoder, self.generator, self.param_extractor, self.discriminator1, self.discriminator2]
        self.optimizer_list = [self.e_optim, self.g_optim, self.p_optim, self.d1_optim, self.d2_optim]
        self.scheduler = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95) for optimizer in self.optimizer_list]

    def init_params(self, cfg):
        for key, value in cfg.items():
            setattr(self, key, value)
        self.patch_size = self.size // 8

    def __call__(self):
        self.prepare_models()
        loader_train = DataLoader(dataset=self.dataset_train, batch_size=self.batch_size, shuffle=True)
        print('%d Train Start' % (self.init_epoch + 1))
        for epoch in range(self.init_epoch + 1, self.end_epoch + 1):
            start_time = time.time()
            epoch_loss = self.init_epoch_loss(epoch)

            for data in loader_train:
                image, info = data[0].to(self.device), data[1]
                self.zero_optimizers()
                self.vector_tensor = param_to_vector(info).to(self.device).to(torch.float32)
                gen_loss = self.train_generator(image)  # Generator Train
                dis_loss = self.train_discriminator(image)  # Discriminator Train
                self.update_epoch_loss(epoch_loss, gen_loss, dis_loss)  # Loss Update

            self.save_epoch_loss(start_time, epoch_loss)
            self.update_schedulers()

            # Save Image
            if epoch % 5 == 0:
                self.evaluate_and_save(epoch)

    def train_generator(self, image):
        self.discriminator1.requires_grad_(False)
        self.discriminator2.requires_grad_(False)
        recon_loss, adv_loss1, adv_loss2, content_loss, param_loss = self.generator_criterion(image)
        g_total_loss = sum(x * y for x, y in zip([recon_loss, adv_loss1, adv_loss2, content_loss, param_loss], list(self.loss_weight.values())[2:]))
        g_total_loss.backward()
        self.e_optim.step()
        self.g_optim.step()
        self.p_optim.step()
        return [recon_loss, adv_loss1, adv_loss2, content_loss, param_loss]

    def train_discriminator(self, image):
        self.discriminator1.requires_grad_(True)
        self.discriminator2.requires_grad_(True)
        d1_loss, d2_loss = self.discriminator_criterion(image)
        d_total_loss = sum(x * y for x, y in zip([d1_loss, d2_loss], list(self.loss_weight.values())[:2]))
        d_total_loss.backward()
        self.d1_optim.step()  # self.d_optim에 대해 step 호출
        self.d2_optim.step()
        return [d1_loss, d2_loss]

    def update_epoch_loss(self, epoch_loss, gen_loss, dis_loss):
        for name, loss in zip(list(self.loss_weight.keys()), dis_loss + gen_loss):
            epoch_loss[name] += loss.item() / self.dataset_train.total_num

    def init_epoch_loss(self, epoch):
        return {'epoch': epoch, **{name: 0 for name in self.loss_weight.keys()}}

    def zero_optimizers(self):
        for optimizer in self.optimizer_list:
            optimizer.zero_grad()

    def evaluate_and_save(self, epoch):
        self.encoder.eval()
        self.generator.eval()
        self.param_extractor.eval()
        with torch.no_grad():
            data = self.dataset_train.load_sample()
            image, info = data[0].to(self.device), data[1]
            self.vector_tensor = param_to_vector(info).to(self.device).to(torch.float32)
            save_img = self.save_image(image)
            self.save_models_and_output(epoch, save_img)
        self.encoder.train()
        self.generator.train()
        self.param_extractor.train()

    def save_models_and_output(self, epoch, save_img):
        create_emtpy_folder([self.path + 'output/', self.path + 'weight/'])
        io.imsave(f'{self.path}output/{epoch}.jpg', np.uint8(save_img * 255))
        for name, model in zip(self.model_names, self.model_list):
            torch.save(model.state_dict(), self.path + 'weight/%s_%d.pt' % (name, epoch))

    def save_epoch_loss(self, start_time, epoch_loss):
        es_time = time.time() - start_time
        epoch_loss['time'] = es_time
        print(', '.join([f'{key}: {value}' for key, value in epoch_loss.items()]))

    def load_model(self):
        for model_name in self.model_names:
            file_path = self.path + 'weight/%s_%d.pt' % (model_name, self.init_epoch)
            getattr(self, model_name).load_state_dict(torch.load(file_path))
        print('%d Model Load Success' % self.init_epoch)

    def prepare_models(self):
        if self.init_epoch != 0:
            self.load_model()
        for model in self.model_list:
            model.train()
        self.init_patch(batch_size=self.batch_size)

    def init_patch(self, batch_size=1):
        self.real_patch = torch.ones(batch_size, 1, self.patch_size, self.patch_size, requires_grad=False).to(self.device)
        self.fake_patch = torch.zeros(batch_size, 1, self.patch_size, self.patch_size, requires_grad=False).to(self.device)

    def generator_criterion(self, image):
        if image.shape[0] != self.real_patch.shape[0]:
            self.init_patch(batch_size=image.shape[0])

        output_list, style_list, content_list = self.middle_output(image)

        recon_loss = torch.mean(torch.stack([self.mae_loss(image[i,], output_list[i][i,]) for i in range(len(output_list))]))

        dis_list1, dis_list2 = self.discriminator_output(output_list)
        adversarial_loss1 = torch.mean(torch.stack([self.mse_loss(self.real_patch, out_fake) for out_fake in dis_list1]))
        adversarial_loss2 = torch.mean(torch.stack([self.mse_loss(self.real_patch, out_fake) for out_fake in dis_list2]))

        re_style_list, re_content_list = zip(*[self.encoder(output, context_vector=False, self_style=True)[:2] for output in output_list])

        content_loss = torch.mean(torch.stack([self.mae_loss(content_list[i], re_content_list[i]) for i in range(len(content_list))]))

        param_loss1 = torch.mean(torch.stack([self.mse_loss(self.vector_tensor, self.param_extractor(out_style.to(self.device))) for out_style in style_list]))
        param_loss2 = torch.mean(torch.stack([self.mse_loss(self.vector_tensor, self.param_extractor(out_style.to(self.device))) for out_style in re_style_list]))
        param_loss3 = torch.mean(torch.stack([self.mse_loss(style_list[i], re_style_list[i]) for i in range(len(re_style_list))]))

        return recon_loss, adversarial_loss1, adversarial_loss2, content_loss, param_loss1 + param_loss2 + param_loss3

    def discriminator_output(self, output_list):
        dis1 = [self.discriminator1(output_images, self.vector_tensor) for output_images in output_list]
        dis2 = [self.discriminator2(output_images) for output_images in output_list]
        return dis1, dis2

    def discriminator_criterion(self, image):
        output_list, _, _ = self.middle_output(image)

        # real
        out_real1 = self.discriminator1(image, self.vector_tensor)
        out_real2 = self.discriminator2(image)
        real_loss1 = self.mse_loss(self.real_patch, out_real1)
        real_loss2 = self.mse_loss(self.real_patch, out_real2)

        # fake
        dis_list1, dis_list2 = self.discriminator_output(output_list)
        fake_loss1 = torch.mean(torch.stack([self.mse_loss(self.fake_patch, out_fake) for out_fake in dis_list1]))
        fake_loss2 = torch.mean(torch.stack([self.mse_loss(self.fake_patch, out_fake) for out_fake in dis_list2]))

        return real_loss1 + fake_loss1, real_loss2 + fake_loss2

    def middle_output(self, image):
        output_list, style_list, content_list = [], [], []
        for i in range(image.shape[0]):
            new_tensor = torch.cat((self.vector_tensor[:i], self.vector_tensor[i + 1:]), dim=0)
            _image = image[[i],].repeat(image.shape[0] - 1, 1, 1, 1)
            style_code, content_code, mid_layer = self.encoder(_image, new_tensor)

            self_style_code, self_content_code, self_mid_layer = self.encoder(image[[i],], context_vector=False, self_style=True)

            first_part, second_part = style_code[:i], style_code[i:]
            final_style_code = torch.cat((first_part, self_style_code, second_part), dim=0)

            first_part, second_part = content_code[:i], content_code[i:]
            final_content_code = torch.cat((first_part, self_content_code, second_part), dim=0)

            final_mid_layer = []
            for j in range(3):
                first_part, second_part = mid_layer[j][:i], mid_layer[j][i:]
                final_mid_layer.append(torch.cat((first_part, self_mid_layer[j], second_part), dim=0))

            output = self.generator(final_style_code, final_content_code, final_mid_layer)

            output_list.append(output)
            style_list.append(final_style_code)
            content_list.append(final_content_code)
        return output_list, style_list, content_list

    def update_schedulers(self):
        for sch in self.scheduler:
            sch.step()

    def save_image(self, image):
        output_list, _, _ = self.middle_output(image)
        save_list = [concat_images(image[i], output_list[i]) for i in range(len(output_list))]
        save_img = np.float32(torch.concat(save_list, dim=0).cpu().detach().numpy())
        return save_img


def param_to_vector(info):
    vector_list = []
    for _info in info:
        _info = eval(_info)
        vectors = np.array([round(_info['Repetition Time'], 2), round(_info['Echo Time'], 2) * 10, round(_info['Inversion Time'], 2),
                            round(_info['Flip Angle'], 2), _info['Magnetic Field Strength'] * 100]) / 3000
        vector_list.append(vectors)
    vector_list = np.stack(vector_list, axis=0)
    return torch.tensor(vector_list)


def concat_images(image, output_images):
    tmp_tensor = image.squeeze()
    for output in output_images:
        output = img_normal(output.squeeze())
        tmp_tensor = torch.concat([tmp_tensor, output], dim=1)
    return tmp_tensor


def img_normal(img):
    img[img < 0] = 0
    img[img > 1] = 1
    return (img - torch.min(img)) / (torch.max(img) - torch.min(img))
