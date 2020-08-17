import os

import torch
from tqdm import tqdm

import config
from BlackBox.CNN import imagenet32_models


path_to_checkpoints = config.PATH_TO_IMAGENET32_FP_MODELS

if not os.path.exists(path_to_checkpoints):
    os.makedirs(path_to_checkpoints)

models = [
    imagenet32_models.ResNet18,
    imagenet32_models.ResNet34,
    imagenet32_models.ResNet50,
    imagenet32_models.VGG11,
    imagenet32_models.VGG13,
    imagenet32_models.VGG16,
    imagenet32_models.VGG19,
]

paths = [
    path_to_checkpoints + "resnet18.pth",
    path_to_checkpoints + "resnet34.pth",
    path_to_checkpoints + "resnet50.pth",
    path_to_checkpoints + "vgg11.pth",
    path_to_checkpoints + "vgg13.pth",
    path_to_checkpoints + "vgg16.pth",
    path_to_checkpoints + "vgg19.pth",
]

dataset = imagenet32_models.utils.Imagenet32DatasetLoader(validation=False)
results = {}

for mod_func, PATH in zip(models, paths):
    model = mod_func()

    device = "cpu"
    if torch.cuda.device_count() >= 1:
        model = torch.nn.DataParallel(model)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    lr = 0.1
    criterion = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimiser, milestones=[150, 250], gamma=0.1
    )

    model.train()

    for epoch in tqdm(range(350)):
        for i, (images, labels) in enumerate(dataset.trainloader):
            images = images.to(device)
            labels = labels.to(device)

            optimiser.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()
        scheduler.step()

    acc1, acc5 = dataset.test(model)
    results[type(model.module).__name__] = (acc1, acc5)
    print(results)
    model = model.module
    torch.save(model.state_dict(), PATH)
