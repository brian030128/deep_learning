# Implement your ResNet34_UNet model here



from torch import nn
from unet import Up, OutConv, DoubleConv


class Res34Unet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Res34Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Sequential(
            DoubleConv(in_channels=64, out_channels=64),
            DoubleConv(in_channels=64, out_channels=64),
            DoubleConv(in_channels=64, out_channels=64)
        )

        self.conv3 = nn.Sequential(
            DoubleConv(in_channels=64, out_channels=128),
            DoubleConv(in_channels=128, out_channels=128),
            DoubleConv(in_channels=128, out_channels=128),
            DoubleConv(in_channels=128, out_channels=128)
        )

        self.conv4 = nn.Sequential(
            DoubleConv(in_channels=128, out_channels=256),
            DoubleConv(in_channels=256, out_channels=256),
            DoubleConv(in_channels=256, out_channels=256),
            DoubleConv(in_channels=256, out_channels=256),
            DoubleConv(in_channels=256, out_channels=256),
            DoubleConv(in_channels=256, out_channels=256),
        )

        self.conv5 = nn.Sequential(
            DoubleConv(in_channels=256, out_channels=512),
            DoubleConv(in_channels=512, out_channels=512),
            DoubleConv(in_channels=512, out_channels=512),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))


        self.mid = DoubleConv(in_channels=512, out_channels=256)

        self.up1 = Up(256 + 512, 32)
        self.up2 = Up(32 + 256, 32)
        self.up3 = Up(32 + 128, 32)
        self.up4 = Up(32 + 64, 32)
        self.outc = OutConv(64, n_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x1 = self.conv2(x)
        x2 = self.conv3(x1)
        x3 = self.conv4(x2)
        x4 = self.conv5(x3)

        x5 = self.avg_pool(x4)
        x5 = self.mid(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


