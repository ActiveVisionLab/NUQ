import torchvision
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import numpy as np

from BlackBox.CNN.dataloader_template import DataLoader
from BlackBox.CNN import config


class Imagenet32DatasetLoader(DataLoader):
    def __init__(self, validation=True):
        ######
        # Validation = True splits test into validation + test datasets in a 20/80 proportion
        super(Imagenet32DatasetLoader, self).__init__()
        random_seed = 123456
        ## Sets train, validation, test loader for Imagenet32 dataset
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        transform1 = transform_minus_1()

        trainset = ImageNet32Loader(
            config.PATH_TO_IMAGENET32,
            train=True,
            transform=transform_train,
            target_transform=transform1,
        )

        testset = ImageNet32Loader(
            config.PATH_TO_IMAGENET32,
            train=False,
            transform=transform_test,
            target_transform=transform1,
        )

        test_sampler = None
        if validation:
            validset = ImageNet32Loader(
                config.PATH_TO_IMAGENET32,
                train=False,
                transform=transform_test,
                target_transform=transform1,
            )
            ## Splitting Validation and Test Datasets
            ## Using 20/80 split
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


# This is  here because for some reason imagenet loads
# targets from 1 to 1000 instead of 0 to 999
class transform_minus_1:
    def __init__(self):
        pass

    def __call__(self, x):
        return x - 1


# Creating dataloader that is similar to CIFAR10
class ImageNet32Loader(torchvision.datasets.CIFAR10):
    base_folder = "imagenet-32-batches-py"
    url = ""
    filename = ""
    tgz_md5 = ""
    train_list = [
        ["train_data_batch_1", ""],
        ["train_data_batch_2", ""],
        ["train_data_batch_3", ""],
        ["train_data_batch_4", ""],
        ["train_data_batch_5", ""],
        ["train_data_batch_6", ""],
        ["train_data_batch_7", ""],
        ["train_data_batch_8", ""],
        ["train_data_batch_9", ""],
        ["train_data_batch_10", ""],
    ]

    test_list = [["val_data", ""]]

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(ImageNet32Loader, self).__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=False,
        )

    def _check_integrity(self):
        return True

    def _load_meta(self):
        pass
