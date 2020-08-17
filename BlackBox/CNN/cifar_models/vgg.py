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


class VGG(Base):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class Quantized_VGG(Base):
    def __init__(self, vgg_name, bits):
        super(Quantized_VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], bits)
        self.classifier = nn.Linear(512, 10)

    def quantize(self):
        for mod in self.modules():
            if isinstance(mod, DSConv2d):
                mod.quantize()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, bits):
        layers = []
        in_channels = 3
        counter = 0
        number_layers = len([l for l in cfg if l != "M"])
        # print(number_layers)
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                counter += 1  # Notice that first activation is in index 2
                if counter == number_layers:
                    layers += [
                        DSConv2d(
                            in_channels,
                            x,
                            3,
                            32,
                            bits[counter - 1],
                            padding=1,
                            bias=True,
                        ),
                        nn.BatchNorm2d(x),
                        nn.ReLU(inplace=True),
                    ]
                else:
                    layers += [
                        DSConv2d(
                            in_channels,
                            x,
                            3,
                            32,
                            bits[counter - 1],
                            padding=1,
                            bias=True,
                        ),
                        nn.BatchNorm2d(x),
                        nn.ReLU(inplace=True),
                        BFPActivation(bits[counter], 32),
                    ]
                    # nnActivation(bits[counter], 7, 32)]

                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class QUANTIZED_VGG11(Quantized_VGG):
    number_layers = 8

    def __init__(self, bits, pretrained=True):
        assert len(bits) == self.number_layers
        super(QUANTIZED_VGG11, self).__init__("VGG11", bits)
        if pretrained:
            model = VGG11(pretrained=True)
            self.pretrained(model)


class QUANTIZED_VGG13(Quantized_VGG):
    number_layers = 10

    def __init__(self, bits, pretrained=True):
        assert len(bits) == self.number_layers
        super(QUANTIZED_VGG13, self).__init__("VGG13", bits)
        if pretrained:
            model = VGG13(pretrained=True)
            self.pretrained(model)


class QUANTIZED_VGG16(Quantized_VGG):
    number_layers = 13

    def __init__(self, bits, pretrained=True):
        assert len(bits) == self.number_layers
        super(QUANTIZED_VGG16, self).__init__("VGG16", bits)
        if pretrained:
            model = VGG16(pretrained=True)
            self.pretrained(model)


class QUANTIZED_VGG19(Quantized_VGG):
    number_layers = 16

    def __init__(self, bits, pretrained=True):
        assert len(bits) == self.number_layers
        super(QUANTIZED_VGG19, self).__init__("VGG19", bits)
        if pretrained:
            model = VGG19(pretrained=True)
            self.pretrained(model)


class VGG11(VGG):
    number_layers = 8

    def __init__(self, pretrained=False):
        super(VGG11, self).__init__("VGG11")
        if pretrained:
            self.load_state_dict(
                torch.load(config.PATH_TO_CIFAR10_FP_MODELS + "vgg11.pth")
            )


class VGG13(VGG):
    number_layers = 10

    def __init__(self, pretrained=False):
        super(VGG13, self).__init__("VGG13")
        if pretrained:
            self.load_state_dict(
                torch.load(config.PATH_TO_CIFAR10_FP_MODELS + "vgg13.pth")
            )


class VGG16(VGG):
    number_layers = 13
    top1 = 0.9388
    top5 = 0.9979

    def __init__(self, pretrained=False):
        super(VGG16, self).__init__("VGG16")
        if pretrained:
            self.load_state_dict(
                torch.load(config.PATH_TO_CIFAR10_FP_MODELS + "vgg16.pth")
            )


class VGG19(VGG):
    number_layers = 16
    top1 = 0.9374
    top5 = 0.9975

    def __init__(self, pretrained=False):
        super(VGG19, self).__init__("VGG19")
        if pretrained:
            self.load_state_dict(
                torch.load(config.PATH_TO_CIFAR10_FP_MODELS + "vgg19.pth")
            )

