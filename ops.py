import torch
import torch.nn as nn
import torchvision

#conv2d
def conv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = True)

#conv3d
def conv3d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1):
    return nn.Conv3d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = True)

#deconv2d
def deconv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = True)

#deconv3d
def deconv3d(in_channels, out_channels, kernel_size = 4, sride = 2, padding = 1):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = True)

#batchnorm
def batchNorm4d(num_features, eps = 1e-5): #input: N, C, H, W
    return nn.BatchNorm2d(num_features, eps = eps)

def batchNorm5d(num_features, eps = 1e-5): #input: N, C, D, H, W
    return nn.BatchNorm3d(num_features, eps = eps)

#relu
def relu(inplace = True):
    return nn.ReLU(inplace)

def lrelu(negative_slope = 0.2, inplace = True):
    return nn.LeakyReLU(negative_slope, inplace)

