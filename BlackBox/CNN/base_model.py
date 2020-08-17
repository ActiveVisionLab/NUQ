import torch.nn as nn
from BlackBox.Quantization.DSConv.nn.dsconv2d import DSConv2d


class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()

    @classmethod
    def get_number_layers(self):
        return self.number_layers

    @classmethod
    def get_original_accuracy(self):
        return self.top1, self.top5

    def quantize(self):
        for mod in self.modules():
            if isinstance(mod, DSConv2d):
                mod.quantize()

    def pretrained(self, model):
        listm = [
            m
            for m in model.modules()
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d))
        ]
        listb = [
            b
            for b in self.modules()
            if isinstance(b, (DSConv2d, nn.Linear, nn.BatchNorm2d))
        ]

        for i, (modf, modt) in enumerate(zip(listm, listb)):
            if isinstance(modf, nn.Conv2d):
                modt.weight.data = modf.weight.data
                if modf.bias is not None:
                    modt.bias.data = modf.bias.data

            if isinstance(modf, nn.Linear):
                modt.weight.data = modf.weight.data
                modt.bias.data = modf.bias.data

            if isinstance(modf, nn.BatchNorm2d):
                modt.weight.data = modf.weight.data
                modt.bias.data = modf.bias.data
                modt.running_mean.data = modf.running_mean.data
                modt.running_var.data = modf.running_var.data
