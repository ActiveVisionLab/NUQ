import torch
from tqdm import tqdm


class DataLoader:
    def __init__(self):
        self.testloader = None
        self.trainloader = None
        self.validloader = None

    def train(self, model, epochs, initial_lr=0.001):
        ## initial_lr is the last lr of the training script

        model.train()

        if not hasattr(model, "module"):
            if torch.cuda.device_count() >= 1:
                model = torch.nn.DataParallel(model)
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            else:
                device = "cpu"
            model.to(device)
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        criterion = torch.nn.CrossEntropyLoss()
        optimiser = torch.optim.SGD(
            model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=5e-4
        )

        for epoch in tqdm(range(epochs)):
            running_loss = 0.0
            for i, (images, labels) in enumerate(tqdm(self.trainloader)):
                images = images.to(device)
                labels = labels.to(device)

                optimiser.zero_grad()

                model.module.quantize()

                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimiser.step()

                running_loss += loss.item()

                if i % 100 == 0 and i != 0:
                    tqdm.write(
                        "[%d, %d] loss: %.5f" % (epoch + 1, i + 1, running_loss / 500)
                    )
                    running_loss = 0.0

        return model

    def test(self, model):
        return self.__test_val__(model, "test")

    def validate(self, model):
        return self.__test_val__(model, "val")

    def __test_val__(self, model, val_or_train="val"):
        loader = self.testloader if val_or_train == "test" else self.validloader
        model.eval()

        if not hasattr(model, "module"):
            if torch.cuda.device_count() >= 1:
                model = torch.nn.DataParallel(model)
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            else:
                device = "cpu"
            model.to(device)
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model.module.quantize()
        total, correct1, correct5 = 0, 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                sorted_logits = torch.argsort(outputs.data, 1, True)[:, :5]
                total += labels.size(0)
                correct1 += (predicted == labels).sum().item()
                correct5 += (sorted_logits == labels.unsqueeze(1)).sum().item()

        accuracy1 = correct1 / total
        accuracy5 = correct5 / total

        return (accuracy1, accuracy5)
