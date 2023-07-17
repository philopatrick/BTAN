from audioop import bias
import torch
from torch import nn as nn

from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer
from basicsr.utils.registry import ARCH_REGISTRY


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True), nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0), nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y
class Basicbl(nn.Module):
    def __init__(self):
        super(Basicbl,self).__init__()
        self.body = nn.Sequential(nn.Conv2d(64,64,3,1,1,bias=True),nn.ReLU(inplace=True),nn.Conv2d(64,64,3,1,1,bias=True),ChannelAttention(64))

    def forward(self,x):
        res = self.body(x)
        return x + res

class BasicU(nn.Module):
    def __init__(self):
        super(BasicU,self).__init__()
        m = []
        for i in range(6):
            m.append(Basicbl())

        self.body = nn.Sequential(*m)
    def forward(self,x):
        res = self.body(x)
        return x + res

class DenseReca(nn.Module):
    def __init__(self):
        super(DenseReca,self).__init__()
        # self.conv1 = nn.Conv2d(64,64,3,1,1,bias=True)
        # self.conv2 = nn.Conv2d(64,64,3,1,1,bias=True)
        # self.conv3 = nn.Conv2d(64,64,3,1,1,bias=True)
        # self.conv4 = nn.Conv2d(64,64,3,1,1,bias=True)
        # self.ca1 = ChannelAttention(64)
        # self.ca = ChannelAttention(64)
        # self.relu = nn.ReLU(inplace=False)
        # self.conv= nn.Conv2d(64,64,3,1,1,bias=True)
        m = []
        for i in range(20):
            m.append(Basicbl())
        self.body = nn.Sequential(*m)
        self.conv = nn.Conv2d(64,64,3,1,1,bias=True)
    def forward(self,x):
        # res = x
        # x = self.conv1(self.relu(x))
        # x = self.conv2(self.relu(x))
        # x = self.ca1(self.relu(x))
        # res2 = res + x
        # x = self.conv3(self.relu(res2))
        # x2 = self.conv4(self.relu(x))
        # x = self.ca2(self.relu(x2))
        # ress = x
        # for i in range(5):
        #     res = x
        #     x = self.conv(x)
        #     x = self.conv(self.relu(x))
        #     x = self.ca(x)
        #     x+=res
        #x7 = self.conv(x + x1 + x2 + x3 + x4 + x5 + self.relu(x6))
        res = self.conv(self.body(x))
        return x + res
class RIR1(nn.Module):
    def __init__(self):
        super(RIR1,self).__init__()
        m = []
        for i in range(3):
            m.append(BasicU())
        self.body = nn.Sequential(*m)
        self.conv = nn.Conv2d(64,64,3,1,1)

    def forward(self,x):
        res = self.conv(self.body(x))
        return res + x

class RIR2(nn.Module):
    def __init__(self):
        super(RIR2,self).__init__()
        m = []
        for i in range(3):
            m.append(RIR1())
        self.body = nn.Sequential(*m)
        self.conv = nn.Conv2d(64,64,3,1,1)

    def forward(self,x):
        res = self.conv(self.body(x))
        return res + x

class RIR3(nn.Module):
    def __init__(self):
        super(RIR3,self).__init__()
        m = []
        for i in range(3):
            m.append(RIR2())
        self.body = nn.Sequential(*m)
        self.conv = nn.Conv2d(64,64,3,1,1)

    def forward(self,x):
        res = self.conv(self.body(x))
        return res + x

class RIRbody(nn.Module):
    def __init__(self):
        super(RIRbody,self).__init__()
        self.body1 = RIR3()
        # self.body2 = RIR3()
        self.conv1 = nn.Conv2d(64,64,3,1,1)
        # self.conv2 = nn.Conv2d(64,64,3,1,1)
    def forward(self,x):
        res1 = self.conv1(self.body1(x))
        res1+=x
        #res2 = self.conv2(self.body2(res1))
       # res2+=res1

        return res1






@ARCH_REGISTRY.register()
class EDSR(nn.Module):
    """EDSR network structure.

    Paper: Enhanced Deep Residual Networks for Single Image Super-Resolution.
    Ref git repo: https://github.com/thstkdgus35/EDSR-PyTorch

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_block (int): Block number in the trunk network. Default: 16.
        upscale (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_block=16,
                 upscale=4,
                 res_scale=1,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(EDSR, self).__init__()

        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.relu = nn.ReLU(inplace=False)


        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        body = []
        for i in range(10):
            dense = DenseReca()
            body.append(dense)
        self.body = nn.Sequential(*body)
        #self.body = RIRbody()
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.shrine = nn.Conv2d(640,64,kernel_size=1)
    def forward(self, x):
        self.mean = self.mean.type_as(x)

        x = (x - self.mean) * self.img_range
        x = self.conv_first(x)
        ress = x
        result = []
        # #resultt = [] python basicsr/train.py -opt options/train/EDSR/train_EDSR_Lx4.yml
        for count,layer in enumerate(self.body):

            x = layer(x)
            result.append(x)
            # if(count<5):
            #     x = layer(x)
            #     result.append(x)
            #     #resultt.append(x)
            # #  print('run to here')
            # else:
            #     x = result[-(count-4)] + x
            #     x = layer(x)
                #resultt.append(x)

        resin = self.shrine(torch.cat(result,1))
        x+=resin
        res = self.conv_after_body(x)
        res += ress

        x = self.conv_last(self.upsample(res))
        x = x / self.img_range + self.mean

        return x
