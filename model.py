import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class LayerActivation:
    features = None

    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()


class Encoder(nn.Module):
    def __init__(self, device):
        super(Encoder, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        self.device = device
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.skip1 = LayerActivation(self.vgg.features, 3)
        self.skip2 = LayerActivation(self.vgg.features, 8)
        self.skip3 = LayerActivation(self.vgg.features, 15)
        self.skip4 = LayerActivation(self.vgg.features, 22)
        self.skip5 = LayerActivation(self.vgg.features, 29)

    def forward(self, x):
        self.vgg(x)

        return x, self.skip1.features.to(self.device), self.skip2.features.to(self.device) \
            , self.skip3.features.to(self.device), self.skip4.features.to(self.device), self.skip5.features.to(self.device),


def upsample(x, convT, skip, conv1x1, device):
    x = convT(x)
    bn = nn.BatchNorm2d(x.shape[1]).to(device)
    x = bn(x)
    x = F.leaky_relu(x, 0.2)

    skip = torch.log(skip ** 2 + 1.0/255.0)
    x = torch.cat([x, skip], dim=1)
    x = conv1x1(x)
    return x


def upsample_last(x, conv1x1_64_3, skip, conv1x1_6_3, device):
    x = conv1x1_64_3(x)
    bn = nn.BatchNorm2d(x.shape[1]).to(device)
    x = bn(x)
    x = F.leaky_relu(x, 0.2)

    skip = torch.log(skip ** 2 + 1.0 / 255.0)
    x = torch.cat([x, skip], dim=1)
    x = conv1x1_6_3(x)
    return x


class Decoder(nn.Module):
    def __init__(self, device):
        super(Decoder, self).__init__()
        self.device = device
        self.latent_representation = nn.Sequential(
            Conv2d(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            Conv2d(512, 512, kernel_size=3, padding=1)
        )
        self.convTranspose_5 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.conv1x1_5 = Conv2d(1024, 512, kernel_size=1)

        self.convTranspose_4 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.conv1x1_4 = Conv2d(1024, 512, kernel_size=1)

        self.convTranspose_3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.conv1x1_3 = Conv2d(512, 256, kernel_size=1)

        self.convTranspose_2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv1x1_2 = Conv2d(256, 128, kernel_size=1)

        self.convTranspose_1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv1x1_1 = Conv2d(128, 64, kernel_size=1)

        self.conv1x1_64_3 = Conv2d(64, 3, kernel_size=1)
        self.conv1x1_6_3 = Conv2d(6, 3, kernel_size=1)

    def forward(self, skip0, skip1, skip2, skip3, skip4, skip5):
        x = self.latent_representation(skip5)
        x = upsample(x, self.convTranspose_5, skip5, self.conv1x1_5, self.device)
        x = upsample(x, self.convTranspose_4, skip4, self.conv1x1_4, self.device)
        x = upsample(x, self.convTranspose_3, skip3, self.conv1x1_3, self.device)
        x = upsample(x, self.convTranspose_2, skip2, self.conv1x1_2, self.device)
        x = upsample(x, self.convTranspose_1, skip1, self.conv1x1_1, self.device)
        x = upsample_last(x, self.conv1x1_64_3, skip0, self.conv1x1_6_3, self.device)
        return x


class Model(nn.Module):
    def __init__(self, device):
        super(Model, self).__init__()
        self.encoder = Encoder(device)
        self.decoder = Decoder(device)

    def forward(self, x):
        x = x.float()
        skip0, skip1, skip2, skip3, skip4, skip5 = self.encoder(x)
        x = self.decoder(skip0, skip1, skip2, skip3, skip4, skip5)
        return x
