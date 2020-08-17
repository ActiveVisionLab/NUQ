import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from tqdm import tqdm

from BlackBox.CNN.dataloader_template import DataLoader
from BlackBox.CNN import config


class CifarDatasetLoader(DataLoader):
    def __init__(self, validation=True):
        ######
        # Validation = True splits test into validation + test datasets in a 20/80 proportion
        super(CifarDatasetLoader, self).__init__()
        random_seed = 123456
        ## Sets train, validation, test loader for Cifar10 dataset
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010],
        )

        crop = transforms.RandomCrop(32, padding=4)
        augment = transforms.RandomHorizontalFlip()
        toTensor = transforms.ToTensor()

        transform_train = transforms.Compose([crop, augment, toTensor, normalize])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        trainset = torchvision.datasets.CIFAR10(
            root=config.PATH_TO_CIFAR10,
            train=True,
            download=True,
            transform=transform_train,
        )

        testset = torchvision.datasets.CIFAR10(
            root=config.PATH_TO_CIFAR10,
            train=False,
            download=True,
            transform=transform_test,
        )

        ### In case we want to just train the model instead of doing model selection, validation = False
        test_sampler = None
        if validation:
            validset = torchvision.datasets.CIFAR10(
                root=config.PATH_TO_CIFAR10,
                train=False,
                download=True,
                transform=transform_test,
            )

            ## Splitting Validation and Test Datasets
            ## Using 20/80 split, which means
            # 2000 images to Validation Set
            # 8000 images to Test Set
            # This is likely representative, since for 10 classes, each will have 800 images for testing
            num_train = len(testset)
            indices = list(range(num_train))
            split = int(np.floor(0.2 * num_train))

            np.random.seed(random_seed)
            np.random.shuffle(indices)

            test_idx, valid_idx = indices[split:], indices[:split]
            test_sampler = SubsetRandomSampler(test_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)

            self.validloader = torch.utils.data.DataLoader(
                validset, batch_size=128, sampler=valid_sampler, num_workers=8
            )

        self.trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=8
        )

        self.testloader = torch.utils.data.DataLoader(
            testset, batch_size=256, sampler=test_sampler, num_workers=8
        )
