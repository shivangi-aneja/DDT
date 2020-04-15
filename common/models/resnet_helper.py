"https://github.com/kevinlu1211/pytorch-unet-resnet-50-encoder/blob/master/u_net_resnet_50_encoder.py"
import torch
from torch import nn

class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, nonlinearity=nn.ReLU()):
        super(ConvBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.nonlinearity = nonlinearity
        nn.init.xavier_normal_(self.conv.weight.data)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.nonlinearity(x)
        return x


class UpBlockResNet(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, non_linearity=nn.ReLU()):
        super(UpBlockResNet, self).__init__()

        # if upsampling_method == "conv_transpose":
        # self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # elif upsampling_method == "bilinear":
        #     self.upsample = nn.Sequential(
        #         nn.Upsample(mode='bilinear', scale_factor=2),
        #         nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        #     )
        self.conv_block_1 = ConvBlock(in_channels, out_channels, nonlinearity=non_linearity)
        # self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        # x = self.upsample(x)
        # x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        # x = self.conv_block_2(x)
        return x


class Decoder1:
    def __init__(self, latent_dim):

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 1, 0, bias=False),  # Output HxW = 4x4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # Output HxW = 8x8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # Output HxW = 16x16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),  # Output HxW = 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),  # Output HxW = 64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),  # Output HxW = 128x128
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(16, 3, 4, 2, 1, bias=False),  # Output HxW = 256x256
            nn.Tanh()
        )