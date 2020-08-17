"""ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# from DSConv.DSConv2d import DSConv2d
# from DSConv.Activation import nnActivation
from BlackBox.Quantization.DSConv.nn.dsconv2d import DSConv2d
from BlackBox.Quantization.src.bfpactivation import BFPActivation
from BlackBox.CNN.base_model import Base
from BlackBox.CNN import config


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(Base):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet18(ResNet):
    number_layers = 20
    top1 = 0.9541
    top5 = 0.9986

    def __init__(self, pretrained=False):
        super(ResNet18, self).__init__(BasicBlock, [2, 2, 2, 2])
        if pretrained:
            self.load_state_dict(
                torch.load(config.PATH_TO_CIFAR10_FP_MODELS + "resnet18.pth")
            )


class ResNet34(ResNet):
    number_layers = 36

    def __init__(self, pretrained=False):
        super(ResNet34, self).__init__(BasicBlock, [3, 4, 6, 3])
        if pretrained:
            self.load_state_dict(
                torch.load(config.PATH_TO_CIFAR10_FP_MODELS + "resnet34.pth")
            )


class ResNet50(ResNet):
    number_layers = 53
    top1 = 0.953
    top5 = 0.9988

    def __init__(self, pretrained=False):
        super(ResNet50, self).__init__(Bottleneck, [3, 4, 6, 3])
        if pretrained:
            self.load_state_dict(
                torch.load(config.PATH_TO_CIFAR10_FP_MODELS + "resnet50.pth")
            )


class ResNet101(ResNet):
    number_layers = 104

    def __init__(self, pretrained=False):
        super(ResNet101, self).__init__(Bottleneck, [3, 4, 23, 3])
        if pretrained:
            self.load_state_dict(
                torch.load(config.PATH_TO_CIFAR10_FP_MODELS + "resnet101.pth")
            )


class ResNet152(ResNet):
    number_layers = 155

    def __init__(self, pretrained=False):
        super(ResNet152, self).__init__(Bottleneck, [3, 8, 36, 3])
        if pretrained:
            self.load_state_dict(
                torch.load(config.PATH_TO_CIFAR10_FP_MODELS + "resnet152.pth")
            )


##############################
# QUANTIZED RESNET
##############################


class QuantBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, bits, stride=1):
        super(QuantBasicBlock, self).__init__()
        first_bit = bits.pop(0)
        self.conv1 = DSConv2d(
            in_planes,
            planes,
            3,
            block_size=32,
            bit=first_bit,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.activation1 = BFPActivation(bits[0], 32)
        self.conv2 = DSConv2d(
            planes,
            planes,
            kernel_size=3,
            block_size=32,
            bit=bits.pop(0),
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        # This is just because the last layer doesn't need to quantize the activation
        self.activation2 = (
            BFPActivation(bits[0], 32) if len(bits) > 0 else nn.Identity()
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                DSConv2d(
                    in_planes,
                    self.expansion * planes,
                    1,
                    block_size=32,
                    bit=first_bit,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.activation1(F.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation2(F.relu(out))
        return out


class QuantBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, bits, stride=1):
        super(QuantBottleneck, self).__init__()
        first_bit = bits.pop(0)
        self.conv1 = DSConv2d(in_planes, planes, 1, 32, bit=first_bit, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.activation1 = BFPActivation(bits[0], 32)
        self.conv2 = DSConv2d(
            planes,
            planes,
            kernel_size=3,
            block_size=32,
            bit=bits.pop(0),
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.activation2 = BFPActivation(bits[0], 32)
        self.conv3 = DSConv2d(
            planes,
            self.expansion * planes,
            kernel_size=1,
            block_size=32,
            bit=bits.pop(0),
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.activation3 = (
            BFPActivation(bits[0], 32) if len(bits) > 1 else nn.Identity()
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                DSConv2d(
                    in_planes,
                    self.expansion * planes,
                    1,
                    block_size=32,
                    bit=first_bit,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.activation1(F.relu(self.bn1(self.conv1(x))))
        out = self.activation2(F.relu(self.bn2(self.conv2(out))))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.activation3(F.relu(out))
        return out


class QUANTIZED_ResNet(Base):
    def __init__(self, block, num_blocks, bits, num_classes=10):
        super(QUANTIZED_ResNet, self).__init__()
        self.in_planes = 64

        _bits_ = bits.copy()

        self.conv1 = DSConv2d(
            3,
            64,
            kernel_size=3,
            block_size=32,
            bit=_bits_.pop(0),
            stride=1,
            padding=1,
            bias=False,
        )
        self.activation = BFPActivation(_bits_[0], 32)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, bits=_bits_)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, bits=_bits_)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, bits=_bits_)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, bits=_bits_)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, bits):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, bits, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.activation(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class QUANTIZED_ResNet18(QUANTIZED_ResNet):
    number_layers = 20

    def __init__(self, bits, pretrained=True):
        super(QUANTIZED_ResNet18, self).__init__(QuantBasicBlock, [2, 2, 2, 2], bits)
        if pretrained:
            model = ResNet18(pretrained=True)
            self.pretrained(model)


class QUANTIZED_ResNet34(QUANTIZED_ResNet):
    number_layers = 36

    def __init__(self, bits, pretrained=True):
        super(QUANTIZED_ResNet34, self).__init__(QuantBasicBlock, [3, 4, 6, 3], bits)
        if pretrained:
            model = ResNet34(pretrained=True)
            self.pretrained(model)


class QUANTIZED_ResNet50(QUANTIZED_ResNet):
    number_layers = 53

    def __init__(self, bits, pretrained=True):
        super(QUANTIZED_ResNet50, self).__init__(QuantBottleneck, [3, 4, 6, 3], bits)
        if pretrained:
            model = ResNet50(pretrained=True)
            self.pretrained(model)


class QUANTIZED_ResNet101(QUANTIZED_ResNet):
    number_layers = 104

    def __init__(self, bits, pretrained=True):
        super(QUANTIZED_ResNet101, self).__init__(QuantBottleneck, [3, 4, 23, 3], bits)
        if pretrained:
            model = ResNet101(pretrained=True)
            self.pretrained(model)


class QUANTIZED_ResNet152(QUANTIZED_ResNet):
    number_layers = 155

    def __init__(self, bits, pretrained=True):
        super(QUANTIZED_ResNet152, self).__init__(QuantBottleneck, [3, 8, 36, 3], bits)
        if pretrained:
            model = ResNet152(pretrained=True)
            self.pretrained(model)


if __name__ == "__main__":
    net = QUANTIZED_ResNet18([8 for _ in range(20)])
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

