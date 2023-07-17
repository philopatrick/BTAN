from traceback import print_tb
import torch
from torch import nn as nn

from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer
from basicsr.utils.registry import ARCH_REGISTRY


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class DenseReca(nn.Module):
    def __init__(self):
        super(DenseReca,self).__init__()
        self.conv = nn.Conv2d(64,64,3,1,1,bias=True)
        #self.convfu = nn.Conv2d(512,96,bias=True, kernel_size=1)
        self.ca = ChannelAttention(64)
        #self.ca2 = ChannelAttention(512)
        self.relu = nn.ReLU(inplace=False)
    def forward(self,x):
        res = x
        x1 = self.conv(x)
        x2 = self.conv(x+self.relu(x1))
        x3 = self.ca(x + x1 + self.relu(x2))
        x4 = self.conv(x + x1 + x2 + self.relu(x3))
        x5 = self.conv(x + x1 + x2 + x3 + self.relu(x4))
        x6 = self.ca(x + x1 + x2 + x3 + x4 + self.relu(x5))
        # x7 = self.conv(x + x1 + x2 + x3 + x4 + x5 +  self.relu(x6))
        # x6 = self.ca2(torch.cat([x1,x2,x4,x5],1))
        # x7 = self.convfu(self.relu(x6))
        #res+= x6
        return res + x6
        # res = x
        # x1 = self.conv(self.relu(x))
        # x2 = self.conv(self.relu(x1))
        # x3 = self.conv(self.relu(x2))
        # x4 = self.conv(self.relu(x3))
        # x5 = torch.cat([x1,x2,x3,x4],1)
        # x5 = self.ca(self.relu(x5))
        # x = self.convfu(self.relu(x5))
        # x  = torch.add(x,res)
        # return x
class RecaBody(nn.Module):
    def __init__(self,basicblock,layer_number):
        super(RecaBody,self).__init__()
        self.body = nn.ModuleList([basicblock() for i in range(layer_number)])

    # 1->39 2->38 19->21 20
    def forward(self,x):
        for layer in self.body:
            x = layer(x)
    #     lists = tuple(x.size())
    #    # print(lists)
    #     data = []
    # #     xshape = (20,) + lists
    # #   #  print(xshape)
    # #     self.data = torch.rand(xshape).cuda()
    #     for count,layer in enumerate(self.body):
    #         if(count<20):
    #             x = layer(x)
    #             data.append(x)
    #           #  print('run to here')
    #         else:
    #             x1 = data[-(count-19)] + x
    #             x = layer(x1)



        return x




@ARCH_REGISTRY.register()
class EDSR(nn.Module):
    """EDSR network structure.

    Paper: Enhanced Deep Residual Networks for Single Image Super-Resolution.
    Ref git repo: https://github.com/thstkdgus35/EDSR-PyTorch

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 80.
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
                 num_feat=80,
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
        for i in range(40):
            dense = DenseReca()
            body.append(dense)
        self.body = nn.Sequential(*body)
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        #self.ca = ChannelAttention(128)
        self.shrine = nn.Conv2d(5120,128,kernel_size=1)
    def forward(self, x):
        self.mean = self.mean.type_as(x)

        x = (x - self.mean) * self.img_range
        x = self.conv_first(x)
        ress = x

        result = []
        #resin = x
        #x = self.body(x)
        #x = self.body(x)
        for count,layer in enumerate(self.body):
           x = layer(x)
           result.append(x)
            # if(count<20):
            #     x = layer(x)
            #     result.append(x)
            #   #  print('run to here')
            # else:
            #     x1 = result[-(count-19)] + x
            #     x = layer(x1)

        #     dataout.append(resin)

            #result.append(x)
        catx = torch.cat(result,1)
       # catx = self.ca(catx)
        xshrine = self.shrine(catx)
        # python basicsr/train.py -opt options/train/EDSR/train_EDSR_Lx4.yml

       # x += xshrine
        #datafange = torch.stack(dataout,dim=0).sum(dim=0)
        #resin+=datafange
        #resin = torch.cat(resin,1)
        #resin = self.shrine(resin)
        resin = xshrine + x
        res = self.conv_after_body(resin)
        res = res +  ress

        x = self.conv_last(self.upsample(res))
        x = x / self.img_range + self.mean

        return x
