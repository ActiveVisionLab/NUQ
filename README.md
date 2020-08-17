# NUQ: Finding Non-Uniform Quantization Schemesusing Multi-Task Gaussian Processes
Implementation of ECCV2020 ["Finding Non-Uniform Quantization Using Multi-Task Gaussian Processes"](https://arxiv.org/abs/2007.07743).

## Requirements

* Python >= 3.6
* PyTorch >= 1.0
* GPyTorch
* Pyro

## Dependencies

This depends on the [Quantization](https://github.com/ActiveVisionLab/Quantization) github repo, which implements cuda version of BFP and [DSConv](https://arxiv.org/abs/1901.01928)

Make sure to `git submodule update --init --recursive` and follow the installation steps in the Quantization repo

## Config

In the config.py file, you should insert the paths to the respective variables.

If you have the models trained for cifar10, and imagenet32, just insert their paths in the `config.py` file. If you don't, then you should insert the paths for the desirable loaction in the `config.py` file, and run the `train_cifar.py` or `train_imagenet32.py` scripts.

## Citation

    @misc{nascimento2020finding,
        title={Finding Non-Uniform Quantization Schemes using Multi-Task Gaussian Processes},
        author={Marcelo Gennari do Nascimento and Theo W. Costain and Victor Adrian Prisacariu},
        year={2020},
        eprint={2007.07743},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }