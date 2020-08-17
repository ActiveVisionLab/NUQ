import torch
import itertools
import numpy as np
import copy
from itertools import product


class Data:
    def __init__(
        self,
        number_tasks,
        dimension,
        grid_x_test_size=30,
        grid_x_train_size=30,
        num_samples=100,
    ):
        self.data = {
            "x": [[] for _ in range(number_tasks)],
            "y": [[] for _ in range(number_tasks)],
        }
        self.total_datapoints = 0
        self.data_points_per_task = [0 for _ in range(number_tasks)]
        self.dim = dimension
        self.number_tasks = number_tasks
        self.grid_x_test_size = grid_x_test_size
        self.grid_x_train_size = grid_x_train_size
        self.num_samples = num_samples  ## In cases where we use MCMC

        a_test = np.linspace(0, 1, grid_x_test_size)
        a_train = np.linspace(0, 1, grid_x_train_size)
        self.x_test = torch.tensor(
            [i for i in product(a_test, repeat=dimension)]
        ).float()
        self.x_train = torch.tensor(
            [i for i in product(a_train, repeat=dimension)]
        ).float()

    def push(self, x, y, i):
        ## The values pushed should be in the format:
        ## [number_points=1, dimensions]
        self.data["x"][i].append(x)
        self.data["y"][i].append(y)
        self.total_datapoints += 1
        self.data_points_per_task[i] += 1

    def calculate_datapoints(self):
        for i in range(len(self.data["x"])):
            self.data_points_per_task[i] = len(self.data["x"][i])
        self.total_datapoints = sum(self.data_points_per_task)

    def get_x(self, l):
        return self.data["x"][l]

    def get_y(self, l):
        return self.data["y"][l]

    def clone(self):
        return deepcopy.copy(self.data)

    def get_tensors(self):
        if list(itertools.chain.from_iterable(self.data["x"])):
            full_train_x = torch.cat(
                list(itertools.chain.from_iterable(self.data["x"]))
            )
            full_train_i = torch.cat(
                [
                    i * torch.ones(len(xtsk), dtype=torch.long)
                    for i, xtsk in enumerate(self.data["x"])
                ]
            )
            full_train_y = torch.cat(
                list(itertools.chain.from_iterable(self.data["y"]))
            )

            return (full_train_x, full_train_i), full_train_y
        else:
            return None, None

    def save(self, path):
        np.save(path, self.data)

    def load(self, path, **kwargs):
        self.data = np.load(path, **kwargs).item()
        self.calculate_datapoints()

    def __str__(self):
        torch.set_printoptions(precision=3)
        np.set_printoptions(precision=3)
        print("--------------------------------------------")
        print("Total number of datapoints:", self.total_datapoints)
        for i in range(len(self.data["x"])):
            print("*********")
            print(
                "Datapoints in Task",
                i,
                ", for a total of",
                self.data_points_per_task[i],
                ":",
            )
            for l in range(len(self.data["x"][i])):
                print(
                    "x:",
                    self.data["x"][i][l][0].numpy(),
                    "y:",
                    self.data["y"][i][l][0].numpy(),
                )
        print("--------------------------------------------")
        return ""

