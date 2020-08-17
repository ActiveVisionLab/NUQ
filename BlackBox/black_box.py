import torch


class LinearBlackBox:
    def __init__(self, number_tasks, dimension):
        self.number_tasks = number_tasks
        self.dimension = dimension
        self.slopes = [torch.rand(self.dimension) for _ in range(self.number_tasks)]

    def __call__(self, x_values, task_number):
        return self.evaluate(x_values, task_number)

    def evaluate(self, x_values, task_number):
        assert task_number >= 0 and task_number < self.number_tasks
        return torch.matmul(x_values, self.slopes[task_number])
