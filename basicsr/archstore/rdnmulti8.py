import torch
from torch import nn as nn

from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer
from basicsr.utils.registry import ARCH_REGISTRY

class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning

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

        self.G0 = 16
        self.G = 16
        self.D = 8
        self.C = 4
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.sfe1 = nn.Conv2d(3, num_feat, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=3 // 2)
        #self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat, res_scale=res_scale, pytorch_init=True)
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsample = Upsample(upscale, num_feat)
        self.upsample2 = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.conv_last2 = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.conv_lastlast = nn.Conv2d(num_out_ch, num_out_ch, 3, 1, 1)
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )

        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G0, self.G, self.C))
        self.upsampleraw = nn.Upsample(scale_factor=4,mode='nearest')
    def forward(self, x):
        raw = x

        sfe1 = self.sfe1(x)
        raw1 = sfe1
        sfe2 = self.sfe2(sfe1)

        x = sfe2



        local_features = []
        # for i in range(self.D):
        #     x = self.rdbs[i](x)
        #     local_features.append(x)
        #hour_features = []
        for count, layer in enumerate(self.rdbs):
            x = layer(x)
            raw1 = torch.mul(torch.sigmoid(x),raw1) + raw1
            local_features.append(x)



        # raw1 = torch.mul(torch.sigmoid(local_features[0]),raw1) + raw1
        # raw1 = torch.mul(torch.sigmoid(local_features[1]),raw1) + raw1
        # raw1 = torch.mul(torch.sigmoid(local_features[2]),raw1) + raw1
        # raw1 = torch.mul(torch.sigmoid(local_features[3]),raw1) + raw1
        # raw1 = torch.mul(torch.sigmoid(local_features[4]),raw1) + raw1
        # raw1 = torch.mul(torch.sigmoid(local_features[5]),raw1) + raw1
        # raw1 = torch.mul(torch.sigmoid(local_features[6]),raw1) + raw1
        # raw1 = torch.mul(torch.sigmoid(local_features[7]),raw1) + raw1
        x = self.gff(torch.cat(local_features, 1)) # manipulation
        res = self.conv_after_body(raw1)


        x = self.conv_last(self.upsample(x))
        raw1 = self.conv_last2(self.upsample2(res))
        raw = self.upsampleraw(raw)
        x = torch.mul(torch.sigmoid(x),raw) + raw
        x = self.conv_lastlast(x+raw1)


        return x