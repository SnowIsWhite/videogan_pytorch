import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from ops import*

batchSize = 32
loadSize = 128
fineSize = 64
frameSize = 32
lr = 0.0002

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
                conv3d(3, 128),
                lrelu(0.2),
                conv3d(128,256),
                lrelu(0.2),
                batchNorm5d(256, 1e-3),
                conv3d(256,512),
                lrelu(0.2),
                batchNorm5d(512, 1e-3),
                conv3d(512,1024),
                lrelu(0.2),
                batchNorm5d(1024,1e-3),
                )

    def forward(self, x):
        out = model(x)
        return out

class encode_net(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
                conv2d(3,128),
                relu(),
                conv2d(128,256),
                relu(),
                batchNorm4d(256, 1e-3),
                conv2d(256,512),
                relu(),
                batchNorm4d(512,1e-3),
                conv2d(512,1024),
                relu(),
                batchNorm4d(1024,1e-3)
                )
    def forward(self,x):
        out = model(x)
        return out

#deconv2d
class static_net(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
                deconv2d(1024,512),
                relu(),
                batchNorm4d(512),
                deconv2d(512,256),
                relu(),
                batchNorm4d(256),
                deconv2d(256,128),
                relu(),
                batchNorm4d(128),
                deconv2d(128,3)
                )
    def forward(self,x):
        out = model(x)
        return out

#deconv3d
class net_video(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
                deconv3d(1024,1024, kernel_size = (2,1,1)),
                relu(),
                batchNorm5d(1024),
                deconv3d(1024,512),
                relu(),
                batchNorm5d(512),
                deconv3d(512,256),
                relu(),
                batchNorm5d(256),
                deconv3d(256,128),
                relu(),
                batchNorm5d(128)
                )
    def forward(self,x):
        x = x.view(x.size(0), 1024, 1, 4, 4)
        out = self.model(x)
        return out

class mask_net(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
                nn.Sigmoid(),
                deconv3d(128,1)
                )
    def forward(self,x):
        out = self.model(x)
        return out

class fore_net(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
                nn.Tanh(),
                deconv3d(128,3)
                )
    def forward(self,x):
        out = self.model(x)
        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encode_net_ = encode_net()
        self.net_video_ = net_video()
        self.static_net_ = static_net()
        self.mask_net_ = mask_net()
        self.fore_net_ = fore_net()

        self.static= nn.Sequential(
                self.encode_net_,
                self.static_net_
                )
        self.fore = nn.Sequential(
                self.encode_net_,
                self.net_video_,
                self.fore_net_
                )
        self.mask = nn.Sequential(
                self.encode_net_,
                self.net_video_,
                self.mask_net_
                )
    def forward(self,x):
        out_static = static(x) #batch, 3, 64, 64
        out_fore = fore(x) #batch, 3, 32, 64, 64
        out_mask = mask(x) #batch, 1, 32, 64, 64
        gen1 = out_mask.expand_as(out_fore)*out_fore
        gen2 = (torch.ones(out_mask.size(0),1,32,64,64) - out_mask).expand_as(out_fore) * out_static.unsqueeze(2).expand_as(out_fore)
        gen = gen1+gen2
        return gen

discriminator = Discriminator()
generator = Generator()

#loss and optimizer
criterion = nn.CrossEntropyLoss()
d_optim = torch.optim.Adam(discriminator.parameters(), lr=lr)
g_optim = torch.optim.Adam(generator.parameters(), lr=lr)

"""
TODO: regularizer needs to be implemented
"""

#Trainig
#for epoch in range(100):
#    for i, 

            




