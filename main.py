from MultitaskGP.model import MultitaskGPModel
from MultitaskGP.data_structure import Data
from MultitaskGP.low_fidelity import LowFidelityInformation
from MultitaskGP.high_fidelity import HighFidelityUCB
from DecisionProcedure.max_decision import MaxDecision
from BlackBox.ParametricFunctions import BezierLinear, Chebyshev4
from BlackBox.QuantizationFunction import QuantizationFunction
from BlackBox.black_box import LinearBlackBox  # troubleshooting
from Utils.plotting import plot_mcmc, plot  # for 1D case for visualzation

## Importing the Quantized CNNs
from BlackBox.CNN import cifar_models
from BlackBox.CNN import imagenet32_models
from BlackBox.CNN import imagenet_models

import random
import torch
import gpytorch
import pyro
from pyro.infer.mcmc import NUTS, MCMC
from matplotlib import pyplot as plt


number_points = 61
parametric_function = BezierLinear
dimension = parametric_function.get_number_parameters()
number_tasks = 4
costs = [0.1, 0.1, 0.1]
quantized_cnn = cifar_models.QUANTIZED_ResNet18

data = Data(number_tasks, dimension)
# function = LinearBlackBox(number_tasks, dimension)
function = QuantizationFunction(parametric_function, quantized_cnn, "cifar")


## Start with random initial point (x, l)
x = torch.rand(dimension).unsqueeze(0)
l = random.choice([i for i in range(number_tasks - 1)])  # Can't select last one
## Get its y value
y = function(x, l)
data.push(x, y, l)

print(data)

## For number of points
for i in range(number_points):
    ##  Propose another point (x, l) using MultitaskGP
    (full_train_x, full_train_i), full_train_y = data.get_tensors()
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.Positive()
    )
    model = MultitaskGPModel(
        (full_train_x, full_train_i), full_train_y, likelihood, number_tasks, dimension
    )
    model.apply_priors(likelihood)
    model.train(), likelihood.train()

    ##### Attempt at doing MCMC on Multitask setting
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    num_samples = 100
    warmup_steps = 200

    def pyro_model(x, y):
        model.pyro_sample_from_prior()
        output = model(x, full_train_i)
        loss = mll.pyro_factor(output, y)
        return y

    nuts_kernel = NUTS(pyro_model, adapt_step_size=True)
    mcmc_run = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc_run.run(full_train_x, full_train_y)

    model.pyro_load_from_samples(mcmc_run.get_samples())

    model.eval()

    ## At every 5 points, make a High-Fidelity proposal
    if i % 5 == 0 and i != 0:
        high_fidelity_acquisition = HighFidelityUCB()
        new_x = high_fidelity_acquisition(data, model, likelihood)

        y_found = function(new_x, number_tasks - 1)
        data.push(new_x, y_found, number_tasks - 1)

        print(data)
        # plot_mcmc(data, model, likelihood) ## For the 1D case

    ## Low Fidelity Exploration
    else:
        low_fidelity_acquisition = LowFidelityInformation(dimension, number_tasks, data)
        new_x, fidlty = low_fidelity_acquisition(likelihood, model, data.x_test, costs)

        y_found = function(new_x, fidlty)
        data.push(new_x, y_found, fidlty)

        print(data)


################
# Decision Phase
################
#  After the exploration, we train the hyperparameters using mll
# This is maybe NOT the best approach
(full_train_x, full_train_i), full_train_y = data.get_tensors()
likelihood = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.Positive()
)
model = MultitaskGPModel(
    (full_train_x, full_train_i), full_train_y, likelihood, number_tasks, dimension
)
model.initialize_hyperparameters()
model.train(), likelihood.train()

optimizer = torch.optim.Adam(
    [{"params": model.parameters()},], lr=0.1  # Includes GaussianLikelihood parameters
)

mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(50):
    optimizer.zero_grad()
    output = model(full_train_x, full_train_i)
    loss = -mll(output, full_train_y)
    loss.backward()
    print("Iter %d/50 - Loss: %.3f" % (i + 1, loss.item()))
    optimizer.step()

model.eval(), likelihood.eval()
# plot(data, model, likelihood) ## For the 1D case

## Decision Procedure
decision = MaxDecision()
resulting_parameters = decision(data, model, likelihood, function, quantized_cnn)

print(resulting_parameters)
