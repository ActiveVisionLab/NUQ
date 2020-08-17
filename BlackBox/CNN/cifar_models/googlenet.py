"""GoogLeNet with PyTorch."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from BlackBox.Quantization.DSConv.nn.dsconv2d import DSConv2d
from BlackBox.Quantization.src.bfpactivation import BFPActivation
from BlackBox.CNN.base_model import Base
from BlackBox.CNN import config


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class GoogLeNet(Base):
    number_layers = 64
    top1 = 0.955
    top5 = 0.9983

    def __init__(self, pretrained=False):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, 10)

        if pretrained:
            self.load_state_dict(
                torch.load(config.PATH_TO_CIFAR10_FP_MODELS + "googlenet.pth")
            )

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


#######################
# QUANTIZED NETWORK
#######################


class QuantInception(nn.Module):
    def __init__(
        self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, bits
    ):
        super(QuantInception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            DSConv2d(
                in_planes,
                n1x1,
                kernel_size=1,
                block_size=32,
                bit=bits.pop(0),
                bias=True,
            ),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
            BFPActivation(bits[0], 32),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            DSConv2d(
                in_planes,
                n3x3red,
                kernel_size=1,
                block_size=32,
                bit=bits.pop(0),
                bias=True,
            ),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            BFPActivation(bits[0], 32),
            DSConv2d(
                n3x3red,
                n3x3,
                kernel_size=3,
                block_size=32,
                bit=bits.pop(0),
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
            BFPActivation(bits[0], 32),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            DSConv2d(
                in_planes,
                n5x5red,
                kernel_size=1,
                block_size=32,
                bit=bits.pop(0),
                bias=True,
            ),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            BFPActivation(bits[0], 32),
            DSConv2d(
                n5x5red,
                n5x5,
                kernel_size=3,
                block_size=32,
                bit=bits.pop(0),
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            BFPActivation(bits[0], 32),
            DSConv2d(
                n5x5,
                n5x5,
                kernel_size=3,
                block_size=32,
                bit=bits.pop(0),
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            BFPActivation(bits[0], 32),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            DSConv2d(
                in_planes,
                pool_planes,
                kernel_size=1,
                block_size=32,
                bit=bits.pop(0),
                bias=True,
            ),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
            BFPActivation(bits[0], 32) if len(bits) > 0 else nn.Identity(),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class QUANTIZED_GoogLeNet(Base):
    number_layers = 64

    def __init__(self, bits, pretrained=True):
        super(QUANTIZED_GoogLeNet, self).__init__()

        # _bits_ = bits.tolist().copy()
        _bits_ = bits.copy()
        _bits_.append(10)

        self.pre_layers = nn.Sequential(
            DSConv2d(
                3,
                192,
                kernel_size=3,
                block_size=32,
                bit=_bits_.pop(0),
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
            BFPActivation(_bits_[0], 32),
        )

        self.a3 = QuantInception(192, 64, 96, 128, 16, 32, 32, _bits_)
        self.b3 = QuantInception(256, 128, 128, 192, 32, 96, 64, _bits_)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = QuantInception(480, 192, 96, 208, 16, 48, 64, _bits_)
        self.b4 = QuantInception(512, 160, 112, 224, 24, 64, 64, _bits_)
        self.c4 = QuantInception(512, 128, 128, 256, 24, 64, 64, _bits_)
        self.d4 = QuantInception(512, 112, 144, 288, 32, 64, 64, _bits_)
        self.e4 = QuantInception(528, 256, 160, 320, 32, 128, 128, _bits_)

        self.a5 = QuantInception(832, 256, 160, 320, 32, 128, 128, _bits_)
        self.b5 = QuantInception(832, 384, 192, 384, 48, 128, 128, _bits_)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, 10)

        if pretrained:
            model = GoogLeNet(pretrained=True)
            self.pretrained(model)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

