import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, input_ch=1, filter_size=64, kernel_size=3, stride=1, padding=1):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(input_ch, filter_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(num_features=filter_size)
        self.instance_norm = nn.InstanceNorm2d(num_features=filter_size)
        self.leak_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, pool=False, norm=None, activation=None):
        x = self.conv(x)
        if norm == 'Batch':
            x = self.batch_norm(x)
        elif norm == 'Instance':
            x = self.instance_norm(x)
        if activation == 'ReLU':
            x = self.relu(x)
        elif activation == 'Leaky':
            x = self.leak_relu(x)
        if pool:
            x = self.max_pooling(x)
        return x


class StyleEncoder(nn.Module):
    def __init__(self, input_ch=1, style_ch=1024):
        super(StyleEncoder, self).__init__()
        self.basic_conv1 = BasicConv(input_ch=input_ch, filter_size=style_ch // 2, kernel_size=1, stride=256 // 16, padding=0)
        self.basic_conv2 = BasicConv(input_ch=style_ch // 2, filter_size=style_ch, kernel_size=1, stride=2, padding=0)
        self.linear1 = nn.Linear(5, style_ch // 2)
        self.linear2 = nn.Linear(5, style_ch)
        self.ContextAttention = ContextAttention()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(style_ch, style_ch)

    def forward(self, x, context_vector=False, self_style=False):
        if self_style:
            x = self.basic_conv1(x)
            x = self.basic_conv2(x)
        else:
            query = self.basic_conv1(x)
            key = self.linear1(context_vector).unsqueeze(1)
            value = self.linear1(context_vector).unsqueeze(1)
            x = self.ContextAttention(query, key, value)

            query = self.basic_conv2(x)
            key = self.linear2(context_vector).unsqueeze(1)
            value = self.linear2(context_vector).unsqueeze(1)
            x = self.ContextAttention(query, key, value)

        x = self.gap(x)
        x = x.squeeze(2).squeeze(2)
        x = self.fc(x)
        x = nn.functional.normalize(x, dim=1)
        return x


class ContextAttention(nn.Module):
    def __init__(self):
        super(ContextAttention, self).__init__()

    def forward(self, Q, K, V):
        batch_size, _, height, width = Q.shape

        attention_scores = torch.matmul(Q.view(batch_size, -1, height * width).transpose(-2, -1), K.transpose(-2, -1))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(K.size(-1), dtype=torch.float))
        attention_weights = F.softmax(attention_scores, dim=1)

        output = torch.matmul(attention_weights, V).transpose(-2, -1)

        output = output.view(batch_size, -1, height, width)
        return output


class ContentEncoder(nn.Module):
    def __init__(self, input_ch, filter_size):
        super(ContentEncoder, self).__init__()
        self.basic_conv1 = BasicConv(input_ch=input_ch, filter_size=input_ch * 2, kernel_size=3, stride=1)
        self.residual_encoder1 = ResidualEncoder(input_ch=input_ch * 2, filter_size=filter_size * 2)
        self.basic_conv2 = BasicConv(input_ch=filter_size * 2, filter_size=filter_size * 4, kernel_size=3, stride=1)
        self.residual_encoder2 = ResidualEncoder(input_ch=filter_size * 4, filter_size=filter_size * 4)
        self.basic_conv3 = BasicConv(input_ch=filter_size * 4, filter_size=filter_size * 8, kernel_size=3, stride=1)
        self.residual_encoder3 = ResidualEncoder(input_ch=filter_size * 8, filter_size=filter_size * 8)
        self.conv2d = nn.Conv2d(filter_size * 8, 6, kernel_size=1, stride=1, padding='same')
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        mid_layer = []
        mid_layer.append(x)
        x = self.pool(x)
        x = self.basic_conv1(x, norm='Instance', activation='Leaky')
        x = self.residual_encoder1(x)

        mid_layer.append(x)
        x = self.pool(x)
        x = self.basic_conv2(x, norm='Instance', activation='Leaky')
        x = self.residual_encoder2(x)

        mid_layer.append(x)
        x = self.pool(x)
        x = self.basic_conv3(x, norm='Instance', activation='Leaky')
        x = self.residual_encoder3(x)

        x = self.conv2d(x)
        x = self.relu(x)
        return x, mid_layer


class Discriminator1(nn.Module):
    def __init__(self, input_ch=1, filter_size=128, img_size=256):
        super(Discriminator1, self).__init__()
        self.basic_conv1 = BasicConv(input_ch=input_ch + 1, filter_size=filter_size, kernel_size=4, stride=2, padding=1)
        self.basic_conv2 = BasicConv(input_ch=filter_size + filter_size // 2, filter_size=filter_size * 2, kernel_size=4, stride=2, padding=1)
        self.basic_conv3 = BasicConv(input_ch=filter_size * 3, filter_size=filter_size * 4, kernel_size=4, stride=2, padding=1)
        self.conv2d = nn.Conv2d(filter_size * 4, 1, kernel_size=1, stride=1, padding='same')
        self.fc1 = nn.Linear(5, img_size * img_size)
        self.fc2 = nn.Linear(5, (img_size // 2) * (img_size // 2))
        self.fc3 = nn.Linear(5, (img_size // 4) * (img_size // 4))

    def forward(self, x, ori_vector):
        vector = self.fc1(ori_vector)
        vector = vector.view(vector.size(0), 1, x.shape[2], x.shape[3])
        vector = vector.expand(-1, 1, -1, -1)
        x = torch.cat((x, vector), dim=1)
        x = self.basic_conv1(x, norm='Instance', activation='Leaky')

        vector = self.fc2(ori_vector)
        vector = vector.view(vector.size(0), 1, x.shape[2], x.shape[3])
        vector = vector.expand(-1, x.shape[1] // 2, -1, -1)
        x = torch.cat((x, vector), dim=1)
        x = self.basic_conv2(x, norm='Instance', activation='Leaky')

        vector = self.fc3(ori_vector)
        vector = vector.view(vector.size(0), 1, x.shape[2], x.shape[3])
        vector = vector.expand(-1, x.shape[1] // 2, -1, -1)
        x = torch.cat((x, vector), dim=1)
        x = self.basic_conv3(x, norm='Instance', activation='Leaky')

        out_src = self.conv2d(x)
        return out_src


class Discriminator2(nn.Module):
    def __init__(self, input_ch=1, filter_size=128):
        super(Discriminator2, self).__init__()
        self.basic_conv1 = BasicConv(input_ch=input_ch, filter_size=filter_size, kernel_size=4, stride=2, padding=1)
        self.basic_conv2 = BasicConv(input_ch=filter_size, filter_size=filter_size * 2, kernel_size=4, stride=2, padding=1)
        self.basic_conv3 = BasicConv(input_ch=filter_size * 2, filter_size=filter_size * 4, kernel_size=4, stride=2, padding=1)
        self.conv2d = nn.Conv2d(filter_size * 4, 1, kernel_size=1, stride=1, padding='same')

    def forward(self, x):
        x = self.basic_conv1(x, norm='Instance', activation='Leaky')
        x = self.basic_conv2(x, norm='Instance', activation='Leaky')
        x = self.basic_conv3(x, norm='Instance', activation='Leaky')

        out_src = self.conv2d(x)
        return out_src


class ResidualEncoder(nn.Module):
    def __init__(self, input_ch=1, filter_size=32):
        super(ResidualEncoder, self).__init__()
        self.basic_conv1 = BasicConv(input_ch=input_ch, filter_size=filter_size, kernel_size=3, stride=1, padding='same')
        self.basic_conv2 = BasicConv(input_ch=input_ch, filter_size=filter_size, kernel_size=3, stride=1, padding='same')
        self.basic_conv3 = BasicConv(input_ch=input_ch, filter_size=filter_size, kernel_size=3, stride=1, padding='same')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        residual = x
        x = self.basic_conv1(x, norm='Instance', activation='Leaky')
        x = self.basic_conv2(x, norm='Instance', activation='Leaky')
        x = self.basic_conv3(x, norm='Instance', activation='Leaky')
        return residual + x


class ResidualGenerator(nn.Module):
    def __init__(self, input_ch, style_ch, filter_size):
        super(ResidualGenerator, self).__init__()
        self.conv_mod_layer1 = Conv2DMod(in_channels=input_ch, out_channels=filter_size,
                                         kernel_size=3, stride=1, padding='same', style_dim=style_ch)
        self.conv_mod_layer2 = Conv2DMod(in_channels=filter_size, out_channels=filter_size,
                                         kernel_size=3, stride=1, padding='same', style_dim=style_ch)
        self.conv_mod_layer3 = Conv2DMod(in_channels=filter_size, out_channels=filter_size,
                                         kernel_size=3, stride=1, padding='same', style_dim=style_ch)
        self.conv_mod_layer4 = Conv2DMod(in_channels=filter_size, out_channels=filter_size,
                                         kernel_size=3, stride=1, padding='same', style_dim=style_ch)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x, style):
        x = self.conv_mod_layer1(x, style)
        x = self.leaky(x)
        residual = x
        x = self.conv_mod_layer2(x, style)
        x = self.leaky(x)

        x = self.conv_mod_layer3(x, style)
        x = self.leaky(x)

        x = self.conv_mod_layer4(x, style)
        x = self.leaky(x)

        return residual + x


class Encoder(nn.Module):
    def __init__(self, input_ch, filter_size, style_ch):
        super(Encoder, self).__init__()
        self.basic_conv = BasicConv(input_ch=input_ch, filter_size=filter_size, kernel_size=3, stride=1)
        self.residual_encoder = ResidualEncoder(input_ch=filter_size, filter_size=filter_size)
        self.style_encoder = StyleEncoder(input_ch=filter_size, style_ch=style_ch)
        self.content_encoder = ContentEncoder(input_ch=filter_size, filter_size=filter_size)

    def forward(self, x, context_vector=False, self_style=False):
        x = self.basic_conv(x, norm='Instance', activation='Leaky')
        x = self.residual_encoder(x)
        style_code = self.style_encoder(x, context_vector, self_style)
        content_code, mid_layer = self.content_encoder(x)
        mid_layer.reverse()

        return style_code, content_code, mid_layer


class Generator(nn.Module):
    def __init__(self, style_ch, filter_size):
        super(Generator, self).__init__()
        self.residual_generator1 = ResidualGenerator(input_ch=6, style_ch=style_ch, filter_size=filter_size * 4)
        self.residual_generator2 = ResidualGenerator(input_ch=filter_size * 8, style_ch=style_ch,
                                                     filter_size=filter_size * 2)
        self.residual_generator3 = ResidualGenerator(input_ch=filter_size * 4, style_ch=style_ch,
                                                     filter_size=filter_size)
        self.residual_generator4 = ResidualGenerator(input_ch=filter_size * 2, style_ch=style_ch,
                                                     filter_size=filter_size)
        self.conv2d = nn.Conv2d(filter_size, 1, kernel_size=1, stride=1, padding=0)
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, style_code, content_code, mid_layer):
        x = self.residual_generator1(content_code, style_code)
        x = self.up_sample(x)
        x = torch.cat((x, mid_layer[0]), dim=1)
        x = self.residual_generator2(x, style_code)
        x = self.up_sample(x)

        x = torch.cat((x, mid_layer[1]), dim=1)
        x = self.residual_generator3(x, style_code)

        x = self.up_sample(x)
        x = torch.cat((x, mid_layer[2]), dim=1)
        x = self.residual_generator4(x, style_code)
        x = self.conv2d(x)

        return x


class Conv2DMod(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, style_dim):
        super(Conv2DMod, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.style_mapping = nn.Linear(style_dim, 2 * in_channels)

    def forward(self, x, style):
        style_params = self.style_mapping(style)
        style_scale, style_shift = style_params.chunk(2, dim=1)
        style_scale = style_scale.unsqueeze(2).unsqueeze(3)
        style_shift = style_shift.unsqueeze(2).unsqueeze(3)
        normalized = nn.functional.instance_norm(x)
        scaled = normalized * style_scale + style_shift
        conv_result = self.conv(scaled)
        return conv_result


class PramExtractor(nn.Module):
    def __init__(self, style_ch, param_size):
        super(PramExtractor, self).__init__()
        self.fc1 = nn.Linear(style_ch, style_ch // 2)
        self.fc2 = nn.Linear(style_ch // 2, style_ch // 4)
        self.fc3 = nn.Linear(style_ch // 4, style_ch // 8)
        self.fc4 = nn.Linear(style_ch // 8, param_size)
        self.relu = nn.ReLU()

    def forward(self, style):
        style = self.fc1(style)
        style = self.relu(style)
        style = self.fc2(style)
        style = self.relu(style)
        style = self.fc3(style)
        style = self.relu(style)
        style = self.fc4(style)
        return style


class SPCNet(object):
    def __init__(self, device):
        self.encoder = Encoder(input_ch=1, filter_size=64, style_ch=512).to(device)
        self.generator = Generator(style_ch=512, filter_size=64).to(device)
        self.discriminator1 = Discriminator1(input_ch=1, filter_size=256).to(device)
        self.discriminator2 = Discriminator2(input_ch=1, filter_size=256).to(device)
        self.param_extractor = PramExtractor(style_ch=512, param_size=5).to(device)
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.e_optim = torch.optim.Adam(self.encoder.parameters(), lr=0.0003, betas=(0.5, 0.999))
        self.g_optim = torch.optim.Adam(self.generator.parameters(), lr=0.0003, betas=(0.5, 0.999))
        self.d1_optim = torch.optim.Adam(self.discriminator1.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.d2_optim = torch.optim.Adam(self.discriminator2.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.p_optim = torch.optim.Adam(self.param_extractor.parameters(), lr=0.0001, betas=(0.5, 0.999))


if __name__ == '__main__':
    feat_q = torch.zeros(1)  # Example query feature vectors
    feat_k = torch.randn(4, 1024)


