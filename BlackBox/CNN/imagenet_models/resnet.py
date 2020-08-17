import torch
import torch.nn as nn
import torchvision.models

from BlackBox.Quantization.DSConv.nn.dsconv2d import DSConv2d
from BlackBox.Quantization.src.bfpactivation import BFPActivation
from BlackBox.CNN.base_model import Base


####################
# QUANTIZED RESNET
####################


def quant_conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return DSConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        block_size=32,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def quant_conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return DSConv2d(
        in_planes, out_planes, kernel_size=1, block_size=32, stride=stride, bias=False
    )


class QuantBasicBlock(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        bits,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(QuantBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = quant_conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.activation1 = BFPActivation(bits.pop(1), 32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = quant_conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.activation2 = BFPActivation(bits.pop(1), 32)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.activation1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        out = self.activation2(out)

        return out


class QuantBottleneck(nn.Module):
    expansion = 4
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        bits,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(QuantBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = quant_conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.activation1 = BFPActivation(bits.pop(1), 32)
        self.conv2 = quant_conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.activation2 = BFPActivation(bits.pop(1), 32)
        self.conv3 = quant_conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.activation3 = BFPActivation(bits.pop(1), 32)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.activation1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.activation2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        out = self.activation3(out)

        return out


class QUANTIZED_ResNet(Base):
    def __init__(
        self,
        block,
        layers,
        bits,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(QUANTIZED_ResNet, self).__init__()

        _bits_ = bits.copy()

        if type(_bits_) is not list:
            _bits_ = bits.tolist().copy()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = DSConv2d(
            3,
            self.inplanes,
            kernel_size=7,
            block_size=32,
            stride=2,
            padding=3,
            bias=False,
        )
        self.activation = BFPActivation(_bits_.pop(1), 32)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], _bits_)
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            _bits_,
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            _bits_,
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            _bits_,
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, QuantBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, QuantBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, bits, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                quant_conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                bits,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    bits,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.activation(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class QUANTIZED_ResNet18(QUANTIZED_ResNet):
    number_layers = 20

    def __init__(self, bits, pretrained=True):
        super(QUANTIZED_ResNet18, self).__init__(QuantBasicBlock, [2, 2, 2, 2], bits)
        if pretrained:
            model = torchvision.models.resnet18(pretrained=True)
            self.pretrained(model)


class QUANTIZED_ResNet34(QUANTIZED_ResNet):
    number_layers = 36

    def __init__(self, bits, pretrained=True):
        super(QUANTIZED_ResNet34, self).__init__(QuantBasicBlock, [3, 4, 6, 3], bits)
        if pretrained:
            model = torchvision.models.resnet34(pretrained=True)
            self.pretrained(model)


class QUANTIZED_ResNet50(QUANTIZED_ResNet):
    number_layers = 53

    def __init__(self, bits, pretrained=True):
        super(QUANTIZED_ResNet50, self).__init__(QuantBottleneck, [3, 4, 6, 3], bits)
        if pretrained:
            model = torchvision.models.resnet50(pretrained=True)
            self.pretrained(model)

