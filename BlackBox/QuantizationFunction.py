import torch
from BlackBox.CNN.cifar_models.utils import CifarDatasetLoader
from BlackBox.CNN.imagenet32_models.utils import Imagenet32DatasetLoader
from BlackBox.CNN.imagenet_models.utils import ImagenetDatasetLoader


class QuantizationFunction:
    def __init__(
        self, parametric_function, quantized_cnn, dataset="cifar", ftoe=[0, 1, 2, 10],
    ):
        ## This takes a parametric_function as initialization (Bezier, Chebyshev, Sinusoidal, etc)
        ## and a quantized_cnn (resnet, vgg, googlenet, etc)
        self.parametric_function = parametric_function
        self.quantized_cnn = quantized_cnn
        self.fidelity_to_epochs = ftoe

        datasets = {
            "cifar": CifarDatasetLoader,
            "imagenet32": Imagenet32DatasetLoader,
            "imagenet": ImagenetDatasetLoader,
        }

        self.dataset = datasets[dataset]()

    def __param_to_bits__(self, number_layers, min_bits, max_bits, func):
        t = torch.linspace(0, 1, number_layers)
        y = func(t)
        y = torch.clamp(y, 0, 1)
        y = torch.floor((max_bits - min_bits - 1) * y).long()
        possible_bits = torch.tensor([i + min_bits for i in range(max_bits - min_bits)])
        return possible_bits[y]

    def param_to_bits(self, parameter_values):
        number_layers = self.quantized_cnn.get_number_layers()
        function = self.parametric_function(parameter_values[0])
        return self.__param_to_bits__(number_layers, 1, 8, function)

    def __call__(self, parameter_values, fidelity):
        number_layers = self.quantized_cnn.get_number_layers()
        epochs = self.fidelity_to_epochs[fidelity]
        # This gets the x value of the BO (in this case the parameter_values) and the fidelity (epochs)
        ## then it returns the accuracy of the quantized_cnn trained on that parameter_values and in the fidelity chosen
        function = self.parametric_function(parameter_values[0])
        bits = self.__param_to_bits__(number_layers, 1, 8, function).numpy().tolist()

        ## Now the CNN should be trained by the number of epochs using the distribution in bits
        # train CNN
        qcnn = self.quantized_cnn(bits, pretrained=True)
        epochs = self.fidelity_to_epochs[fidelity]

        # Train and evaluate cnn
        qcnn = self.dataset.train(qcnn, epochs=epochs)
        accuracy1, accuracy5 = self.dataset.validate(qcnn)

        accuracy = torch.tensor([accuracy1])
        return accuracy
