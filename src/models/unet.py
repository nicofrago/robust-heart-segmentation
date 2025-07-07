import torch
import torch.nn as nn
import torch.nn.functional as F

class DualConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_layers(x)


class DownsampleBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DualConvBlock(input_channels, output_channels)
        )

    def forward(self, x):
        return self.down_conv(x)

class UpsampleBlock(nn.Module):
    def __init__(self, input_channels, output_channels, use_bilinear=True):
        super().__init__()
        if use_bilinear:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.upsample = nn.ConvTranspose2d(input_channels//2, input_channels//2, 2, stride=2)

        self.conv_block = DualConvBlock(input_channels, output_channels)

    def forward(self, x_high, x_low):
        x_high = self.upsample(x_high)
        height_diff = x_low.size()[2] - x_high.size()[2]
        width_diff = x_low.size()[3] - x_high.size()[3]

        x_high = F.pad(x_high, [width_diff // 2, width_diff - width_diff // 2,
                               height_diff // 2, height_diff - height_diff // 2])
        x = torch.cat([x_low, x_high], dim=1)
        return self.conv_block(x)

class UNet(nn.Module):
    def __init__(self, input_channels, num_classes, use_bilinear=True):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.use_bilinear = use_bilinear

        self.initial_conv = DualConvBlock(input_channels, 64)
        self.down1 = DownsampleBlock(64, 128)
        self.down2 = DownsampleBlock(128, 256)
        self.down3 = DownsampleBlock(256, 512)
        self.down4 = DownsampleBlock(512, 512)
        self.up1 = UpsampleBlock(1024, 256, use_bilinear)
        self.up2 = UpsampleBlock(512, 128, use_bilinear)
        self.up3 = UpsampleBlock(256, 64, use_bilinear)
        self.up4 = UpsampleBlock(128, 64, use_bilinear)
        self.final_conv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x1 = self.initial_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.final_conv(x)
