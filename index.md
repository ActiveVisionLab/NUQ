---
layout: index
title: "NUQ: Finding Non-Uniform Quantization Schemes using Multi-Task Gaussian Processes"
---

This is the landing page for paper **Finding Non-Uniform Quantization Schemes using Multi-Task Gaussian Processes**.

## Abstract

We propose a novel method for neural network quantization that casts the neural architecture search problem as one of hyperparameter search to find non-uniform bit distributions throughout the layers of a CNN. We perform the search assuming a Multi-Task Gaussian Processes prior, which splits the problem to multiple tasks, each corresponding to different number of training epochs, and explore the space by sampling those configurations that yield maximum information. We then show that with significantly lower precision in the last layers we achieve a minimal loss of accuracy with appreciable memory savings. We test our findings on the CIFAR10 and ImageNet datasets using the VGG, ResNet and GoogLeNet architectures.
## Code

The code for reproducing results in the paper can be obtained from the [GitHub repository](https://github.com/ActiveVisionLab/NUQ).

## Citation

BiBTeX:

```
@misc{Gennari2018,
author = {Gennari, Marcelo and Costain, Theo W. and Prisacariu, Victor Adrian},
title = {Finding Non-Uniform Quantization Schemesusing Multi-Task Gaussian Processes},
howpublished = {\url{https://arxiv.org/abs/2007.07743}},
year = {2020},
month = {August}
}
```

Plain text:

Marcelo Gennari, Theo W. Costain, Victor A. Prisacariu, "Finding Non-Uniform Quantization Schemes using Multi-Task Gaussian Processes", in [arXiv:2007.07743](https://arxiv.org/abs/2007.07743), 2020
