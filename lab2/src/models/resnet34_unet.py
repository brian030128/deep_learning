import torch
from torch import nn
from .unet import OutConv, DoubleConv
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity  # residual
        out = self.relu(out)
        
        return out

class ResLayer(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, stride=1):
        super(ResLayer, self).__init__()
        
        layers = []
        # First block may have stride > 1 to downsample
        layers.append(ResBlock(in_channels, out_channels, stride))
        
        # Remaining blocks maintain dimensions
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class Res34Unet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Res34Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = ResLayer(64, 64, blocks=3)  # conv2_x
        self.layer2 = ResLayer(64, 128, blocks=4, stride=2)  # conv3_x
        self.layer3 = ResLayer(128, 256, blocks=6, stride=2)  # conv4_x
        self.layer4 = ResLayer(256, 512, blocks=3, stride=2)  # conv5_x
        
        self.mid = DoubleConv(in_channels=512, out_channels=256)
        
        self.up1 = Up(256, 512, 32)
        self.up2 = Up(32, 256, 32)
        self.up3 = Up(32, 128, 32)
        self.up4 = Up(32, 64, 32)
        
        # Final convolution
        self.final_upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        self.outc = OutConv(32, n_classes)
        
    def forward(self, x):
        # Encoder path with ResNet blocks
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x1 = self.layer1(x)      # 64 channels
        x2 = self.layer2(x1)     # 128 channels
        x3 = self.layer3(x2)     # 256 channels
        x4 = self.layer4(x3)     # 512 channels
        
        mid = self.mid(x4)
        
        # Decoder path
        o = self.up1(mid, x4)
        o = self.up2(o, x3)
        o = self.up3(o, x2)
        o = self.up4(o, x1)
        
        o = self.outc(o)
        logits = self.final_upsample(o)
        
        return logits

class CBAM(nn.Module):
    """ Convolutional Block Attention Module (CBAM) """
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel Attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = avg_out + max_out
        x = x * channel_att
        
        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.conv_spatial(torch.cat([avg_out, max_out], dim=1))
        spatial_att = self.sigmoid(spatial_att)
        
        return x * spatial_att

class Up(nn.Module):
    """ Upscaling then CBAM + BatchNorm + ReLU """
    def __init__(self, in_channels, residual_channels, out_channels, mid_channels=32):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(mid_channels + residual_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.cbam = CBAM(out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Padding for size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.cbam(x)  # Apply CBAM attention
        
        return x