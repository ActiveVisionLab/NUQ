"""VGG11/13/16/19 in Pytorch."""
import torch
import torch.nn as nn

# from DSConv.DSConv2d import DSConv2d
# from DSConv.Activation import nnActivation
from BlackBox.Quantization.DSConv.nn.dsconv2d import DSConv2d
from BlackBox.Quantization.src.bfpactivation import BFPActivation
from BlackBox.CNN.base_model import Base
from BlackBox.CNN import config

intr_cfg = [64, 64, "M", 128, 128, "M"]
cfg = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": intr_cfg + [256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": intr_cfg + [256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": intr_cfg
    + [256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 1000)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for i, x in enumerate(cfg):
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                if i < len(cfg) - 2:
                    layers += [
                        nn.Identity()
                    ]  # This is here in order to be compatible with DSConv
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGG11(VGG):
    def __init__(self, pretrained=False):
        super(VGG11, self).__init__("VGG11")
        if pretrained:
            self.load_state_dict(
                torch.load(config.PATH_TO_IMAGENET32_FP_MODELS + "vgg11.pth")
            )


class VGG13(VGG):
    def __init__(self, pretrained=False):
        super(VGG13, self).__init__("VGG13")
        if pretrained:
            self.load_state_dict(
                torch.load(config.PATH_TO_IMAGENET32_FP_MODELS + "vgg13.pth")
            )


class VGG16(VGG):
    def __init__(self, pretrained=False):
        super(VGG16, self).__init__("VGG16")
        if pretrained:
            self.load_state_dict(
                torch.load(config.PATH_TO_IMAGENET32_FP_MODELS + "vgg16.pth")
            )


class VGG19(VGG):
    def __init__(self, pretrained=False):
        super(VGG19, self).__init__("VGG19")
        if pretrained:
            self.load_state_dict(
                torch.load(config.PATH_TO_IMAGENET32_FP_MODELS + "vgg19.pth")
            )


#############
### QUANTIZED


class QUANTIZED_VGG(Base):
    def __init__(self, vgg_name, bits, number_bits):
        super(QUANTIZED_VGG, self).__init__(bits, number_bits)
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 1000)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        length_config = len(cfg)
        for i, x in enumerate(cfg):
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                bit = self.bits.pop(0)
                layers += [
                    DSConv2d(
                        in_channels, x, kernel_size=3, block_size=32, bit=bit, padding=1
                    ),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                if i < len(cfg) - 2:
                    layers += [BFPActivation(self.bits[0], blk=32)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class QUANTIZED_VGG11(QUANTIZED_VGG):
    number_bits = 8
    top1 = 39.5
    top5 = 64.0

    def __init__(self, bits, pretrained=False):
        super(QUANTIZED_VGG11, self).__init__("VGG11", bits, self.number_bits)
        if pretrained:
            model = VGG11(pretrained=True)
            self.pretrained(model)


class QUANTIZED_VGG13(QUANTIZED_VGG):
    number_bits = 10

    def __init__(self, bits, pretrained=False):
        super(QUANTIZED_VGG13, self).__init__("VGG13", bits, self.number_bits)
        if pretrained:
            model = VGG13(pretrained=True)
            self.pretrained(model)


class QUANTIZED_VGG16(QUANTIZED_VGG):
    number_bits = 13

    def __init__(self, bits, pretrained=False):
        super(QUANTIZED_VGG16, self).__init__("VGG16", bits, self.number_bits)
        if pretrained:
            model = VGG16(pretrained=True)
            self.pretrained(model)


class QUANTIZED_VGG19(QUANTIZED_VGG):
    number_bits = 16

    def __init__(self, bits, pretrained=False):
        super(QUANTIZED_VGG19, self).__init__("VGG19", bits, self.number_bits)
        if pretrained:
            model = VGG19(pretrained=True)
            self.pretrained(model)
