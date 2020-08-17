import torch
from prettytable import PrettyTable
import operator
import sys

sys.path.append("BlackBox/Quantization")
from DSConv.nn.dsconv2d import DSConv2d
import numpy as np


class MaxDecision:
    def __init__(self):
        pass

    def memory(self, qcnn, bits):
        mem = 0
        bits = [int(b) for b in bits]
        model = qcnn(bits, pretrained=False)
        for i, mod in enumerate(model.modules()):
            if isinstance(mod, DSConv2d):
                total_size = 1
                for size in mod.weight.shape:
                    total_size *= size
                mem += total_size * mod.bit

        return mem / (8 * 1000 * 1000)

    def __call__(self, data, model, likelihood, function, qcnn):
        ## This will just return the x_values that gives the maximum value of the mean
        test_x = data.x_test
        test_i_l = torch.full(
            (test_x.shape[0],), fill_value=data.number_tasks - 1, dtype=torch.long
        )
        with torch.no_grad():
            f = likelihood(model(test_x, test_i_l))

        mean = f.mean

        ## This is only in case we do MCMC on hyperparameters
        if len(mean.shape) > 1:
            mean = mean.mean(axis=0)

        bits_tested = [
            [function.param_to_bits(x.unsqueeze(0)), [round(i.item(), 3) for i in x]]
            for x in test_x
        ]

        results = {}
        for bits, accuracy_mean in zip(bits_tested, mean):
            str_cnn = "".join(str(e.item()) for e in bits[0])
            str_param = ", ".join(str(e) for e in bits[1])
            if (str_cnn not in results) or results[str_cnn][0] < accuracy_mean:
                results[str_cnn] = [accuracy_mean, str_param]

        table = PrettyTable()

        table.field_names = ["Bits", "Parameters", "Accuracy", "Score", "Memory"]
        res_tb = np.array(
            [],
            dtype=[
                ("Bits", "U100"),
                ("Parameters", "U100"),
                ("Accuracy", float),
                ("Score", float),
                ("Memory", float),
            ],
        )

        for bits in sorted(results, key=results.get, reverse=False):
            score = sum(int(digit) for digit in bits)
            score = score - 4 * len(bits)
            score = results[bits][0] - ((score) / 1000)
            score = round(score.item(), 3)
            acc = round(results[bits][0].item(), 2)
            res_tb = np.append(
                res_tb,
                np.array(
                    (bits, results[bits][1], acc, score, self.memory(qcnn, bits)),
                    dtype=res_tb.dtype,
                ),
            )

        tcopy = res_tb[["Score", "Memory"]].copy()
        tcopy["Memory"] *= -1
        I = np.argsort(tcopy, order=["Score", "Memory"])

        res_tb = res_tb[I]

        for res in res_tb:
            table.add_row(res)

        print(table)
        return res_tb[-1]
