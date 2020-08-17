import torch
import torchvision
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import numpy as np

from BlackBox.CNN.dataloader_template import DataLoader
from BlackBox.CNN import config


class ImagenetDatasetLoader(DataLoader):
    def __init__(self, validation=True):
        ######
        # Validation = True splits test into validation + test datasets in a 20/80 proportion
        super(ImagenetDatasetLoader, self).__init__()
        random_seed = 123456

        ## Sets train, validation, test loader for Imagenet dataset
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        trainset = datasets.ImageFolder(
            config.PATH_TO_IMAGENET + "ILSVRC2012_train", transform_train
        )

        testset = datasets.ImageFolder(
            config.PATH_TO_IMAGENET + "ILSVRC2012_test", transform_test
        )

        test_sampler = None
        if validation:
            validset = datasets.ImageFolder(
                config.PATH_TO_IMAGENET + "ILSVRC2012_test", transform_test
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
