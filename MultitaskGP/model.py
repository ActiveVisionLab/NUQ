import torch
import gpytorch
from gpytorch.priors import LogNormalPrior, NormalPrior, UniformPrior


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks, num_dim):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.num_tasks = num_tasks
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=num_dim)
        )

        # We learn an IndexKernel for 2 tasks
        # (so we'll actually learn 2x2=4 tasks with correlations)
        self.task_covar_module = gpytorch.kernels.IndexKernel(
            num_tasks=num_tasks, rank=num_tasks
        )

    def initialize_hyperparameters(self):
        ## This is a random initialization assuming that the tasks are positively correlated
        hypers = {
            "task_covar_module.covar_factor": torch.rand(
                (self.num_tasks, self.num_tasks)
            )
        }

        self.initialize(**hypers)

    def apply_priors(self, likelihood):
        self.mean_module.register_prior("mean_prior", UniformPrior(0, 1), "constant")
        self.covar_module.base_kernel.register_prior(
            "lengthscale_prior", UniformPrior(0.05, 0.5), "lengthscale"
        )
        self.covar_module.register_prior(
            "outputscale_prior", UniformPrior(1, 2), "outputscale"
        )
        self.task_covar_module.register_prior(
            "covar_factor_prior", UniformPrior(0, 1), "covar_factor"
        )
        self.task_covar_module.register_prior(
            "var_prior", UniformPrior(0.05, 0.3), "var"
        )
        likelihood.register_prior("noise_prior", UniformPrior(0.05, 0.3), "noise")

    def forward(self, x, i):
        mean_x = self.mean_module(x)

        # Get input-input covariance
        covar_x = self.covar_module(x)
        # Get task-task covariance
        covar_i = self.task_covar_module(i)
        # Multiply the two together to get the covariance we want
        covar = covar_x.mul(covar_i)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)
