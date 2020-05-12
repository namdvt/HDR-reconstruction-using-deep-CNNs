import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as st


def get_lum(img):
    luminance = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]

    return luminance


def get_gaussian_kernel(size, sigma=2):
    interval = (2 * sigma + 1.) / size
    x = np.linspace(-sigma - interval / 2., sigma + interval / 2., size + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()

    return nn.Parameter(torch.tensor([[kernel]])).float()


class HDRLoss(nn.Module):
    def __init__(self, device, eps=1.0 / 255.0, tau=0.95, weight=0.5):
        super(HDRLoss, self).__init__()
        self.eps = eps
        self.tau = tau
        self.weight = weight
        self.gaussian_kernel = get_gaussian_kernel(size=5).to(device)

    def forward(self, inputs, outputs, targets, separate_loss=False):
        batch_size, _, width, height = outputs.shape
        total_loss = 0.0
        inputs = inputs.float()

        for b in range(batch_size):
            # compute blending value alpha
            max_c = inputs[b].max(dim=0).values - self.tau
            max_c[max_c < 0] = 0
            alpha = (max_c / (1 - self.tau)).float()

            if separate_loss:
                # target: linear HDR image
                target_luminance = get_lum(targets[b].permute(1, 2, 0))
                target_illumination = F.conv2d(target_luminance.unsqueeze(0).unsqueeze(0), self.gaussian_kernel,
                                               padding=2).squeeze()
                target_reflectance = targets[b].permute(1, 2, 0) - target_illumination.repeat(3, 1, 1).permute(1, 2, 0)

                # output: log HDR image
                output_luminance = get_lum(outputs[b].permute(1, 2, 0))
                output_illumination = F.conv2d(output_luminance.unsqueeze(0).unsqueeze(0), self.gaussian_kernel,
                                               padding=2).squeeze()
                output_reflectance = outputs[b].permute(1, 2, 0) - output_illumination.repeat(3, 1, 1).permute(1, 2, 0)

                # compute loss
                illumination_loss = self.weight * torch.mean((alpha * (target_illumination - output_illumination)) ** 2)
                reflectance_loss = (1 - self.weight) * torch.mean(alpha.repeat(3, 1, 1).permute(1, 2, 0) * (target_reflectance - output_reflectance) ** 2)

                total_loss += illumination_loss + reflectance_loss
            else:
                loss_target_output = torch.mean(alpha * (outputs[b] - targets[b]) ** 2)
                loss_input_output = torch.mean(alpha * (outputs[b] - (inputs[b] ** 2)) ** 2)
                total_loss += loss_target_output + loss_input_output

        return total_loss

