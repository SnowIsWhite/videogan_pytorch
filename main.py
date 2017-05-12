import torch
import torchvision
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from ops import*

batchSize = 32
loadSize = 128
fineSize = 64
frameSize = 32
lr = 0.0002

"""
TODO:
    1. Load data
    2. Implement GPU (easy) and check everything's fine
    3. Save generated video files
"""
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
                conv3d(1024,2, (2,4,4), (1,1,1), (0,0,0))
                )

    def forward(self, x):
        out = self.model(x)
        return out

class encode_net(nn.Module):
    def __init__(self):
        super(encode_net, self).__init__()
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
        out = self.model(x)
        return out

#deconv2d
class static_net(nn.Module):
    def __init__(self):
        super(static_net, self).__init__()
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
        out = self.model(x)
        return out

#deconv3d
class net_video(nn.Module):
    def __init__(self):
        super(net_video, self).__init__()
        self.model = nn.Sequential(
                #deconv3d(1024,1024, kernel_size = (2,1,1), stride = 2),
                #relu(),
                #batchNorm5d(1024),
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
        super(mask_net, self).__init__()
        self.model = nn.Sequential(
                nn.Sigmoid(),
                deconv3d(128,1)
                )
    def forward(self,x):
        out = self.model(x)
        return out

class fore_net(nn.Module):
    def __init__(self):
        super(fore_net, self).__init__()
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
      
    def forward(self,x):
        print("Enter G_forward")
        print("Generating out1")
        out1 = self.encode_net_(x)
        print("Generating out_static")
        out_static = self.static_net_(out1)
        print("Generating out2")
        out2 = self.net_video_(out1)
        print("Generating out_fore")
        out_fore = self.fore_net_(out2)
        print("Generating out_mask")
        out_mask = self.mask_net_(out2)
        """
        out_static  #batch, 3, 64, 64
        out_fore    #batch, 3, 32, 64, 64
        out_mask    #batch, 1, 32, 64, 64
        """
        print("Generating gen1")
        gen1 = out_mask.expand_as(out_fore)*out_fore
        print("Generating mul1")
        mul1 =(np.ones_like(out_mask) - out_mask).expand_as(out_fore)
        print("Generating mul2")
        mul2 = out_static.unsqueeze(2).expand_as(out_fore)
        print("Generating gen2")
        gen2 = mul1*mul2
        gen = gen1+gen2
        print("Leave G_forward")
        return gen

discriminator = Discriminator()
generator = Generator()

#conv2d
#loss and optimizer
#loss_function = nn.BCELoss()
loss_function = nn.CrossEntropyLoss()
reg_loss_function = nn.L1Loss()
d_optim = torch.optim.Adam(discriminator.parameters(), lr=lr)
g_optim = torch.optim.Adam(generator.parameters(), lr=lr)

load_data = torch.rand(10,32,3,32,64,64)

#Trainig videos: batch, 
for epoch in range(100):
    for i, videos in enumerate(load_data):
        temp = videos.permute(2,0,1,3,4)
        videos = Variable(videos)
        images = Variable(temp[0]) #batch, first frame
        
        real_labels = Variable(torch.LongTensor(np.ones(batchSize, dtype = int)))
        fake_labels = Variable(torch.LongTensor(np.zeros(batchSize, dtype = int)))
        print("Training..")
        #train discriminator
        print("train discriminator..")
        discriminator.zero_grad()
        outputs = discriminator(videos).squeeze()
        print(outputs.size())
        print(real_labels.size())
        real_loss = loss_function(outputs, real_labels.long())

        real_score = outputs
    
        fake_videos = generator(images) #gets amazingly slow on my labtop
        outputs = discriminator(fake_videos.detach()).squeeze()
        fake_loss = loss_function(outputs, fake_labels)
        fake_score = outputs

        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optim.step()
        
        #train generator
        print("train generator..")
        generator.zero_grad()
        fake_videos = generator(images)
        outputs = discriminator(fake_videos).squeeze()
        g_loss = loss_function(outputs, real_labels.long())
        g_loss.backward()
        g_optim.step()

        #reg loss
        print("reg loss..")
        reg_loss = reg_loss_function(videos[0], fake_videos[0])
        if True:#(i+1)%10 ==0:
            print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f, '
                    'D(x): %2.f, D(G(x)): %.2f'
                    %(epoch, 50, i+1, 500, d_loss.data[0], g_loss.data[0],
                        real_score.data.mean(), fake_score.data.mean()))

            # save data(gif?)


#save model
torch.save(generator.state_dict(), './generator.pkl')
torch.save(discriminator.state_dict(), './discriminator.pkl')
torch.save(encode_net.state_dict(), './encode.pkl')
torch.save(static_net.state_dict(), './static.pkl')
torch.save(net_video.state_dict(), './netvideo.pkl')
torch.save(mask_net.state_dict(), './mask.pkl')
torch.save(fore_net.state_dict(), './fore.pkl')


