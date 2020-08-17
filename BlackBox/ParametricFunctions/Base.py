from abc import ABC, abstractmethod
import numpy as np


class ParametricFunction(ABC):

    def __init__(self, number_parameters, parameter_values):
        self.number_parameters = number_parameters
        self.parameter_values = parameter_values
        if self.number_parameters != 1:
            assert(len(self.parameter_values) == self.number_parameters)
        super().__init__()

    @classmethod
    def get_number_parameters(self):
        return self.number_parameters

    def update(self, parameter_values):
        self.parameter_values = parameter_values
        if self.number_parameters != 1:
            assert(len(self.parameter_values) == self.number_parameters)

    @abstractmethod
    def __call__(self):
        pass


def parametric_to_channel(number_layers, min_channels, max_channels, func):
    t = np.linspace(0, 1, number_layers)
    y = np.asarray(func(t))
    y = np.maximum(0, np.minimum(1, y))  # This makes sure that func is between 0 and 1 - equivalent to clamp
    assert(((min_channels & (min_channels - 1)) == 0) and min_channels != 0)  # Check if min_channels is a power of 2
    assert(((max_channels & (max_channels - 1)) == 0) and max_channels != 0)  # Check if max_channels is a power of 2
    log_min_channel = np.int(np.log2(min_channels))
    log_max_channel = np.int(np.log2(max_channels))

    y = np.floor(y*(log_max_channel-log_min_channel)).astype(np.int)
    possible_channels = np.asarray([2**(log_min_channel + i) for i in range(log_max_channel-log_min_channel+1)])

    return possible_channels[y]


def parametric_to_bits(number_layers, min_bits, max_bits, func):
    t = np.linspace(0, 1, number_layers)
    y = np.asarray(func(t))
    y = np.maximum(0, np.minimum(1, y))
    y = np.floor((max_bits-min_bits-1)*y).astype(np.int)
    possible_bits = np.asarray([i+min_bits for i in range(max_bits-min_bits)])

    return possible_bits[y]

