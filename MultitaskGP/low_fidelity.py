import torch
import random
import numpy as np
import gpytorch
import math
import copy


class LowFidelityInformation:
    def __init__(self, dimension, number_tasks, data):
        self.dimension = dimension
        self.nmb_t = number_tasks

        self.n = data.grid_x_test_size

        self.test_x = data.x_test.detach().clone()
        self.test_i_M = torch.full(
            (self.test_x.shape[0],), fill_value=self.nmb_t - 1, dtype=torch.long
        )
        self.test_x_final = torch.cat((self.test_x, self.test_x))

    def __call__(self, likelihood, model, test_x, costs):
        likelihood.eval()
        model.eval()

        fm = model(self.test_x, self.test_i_M)
        p_fm = likelihood(fm)
        u = torch.cholesky(p_fm.covariance_matrix)
        inv_sigma_square_M = torch.zeros_like(u)

        if len(u.shape) > 2:
            ## For some reason the cholesky_inverse doesn't support the batch dimension, so we need to for loop it
            for i in range(u.shape[0]):
                inv_sigma_square_M[i] = torch.cholesky_inverse(u[i])
        else:
            inv_sigma_square_M = torch.cholesky_inverse(u)
        maximum_information, index_maximum_information, it = None, None, None

        for l, lbd in enumerate(costs):
            with torch.no_grad():
                test_i_l = torch.full(
                    (self.test_x.shape[0],), fill_value=l, dtype=torch.long
                )

                p_fl_given_D = likelihood(model(self.test_x, test_i_l))
                p_fl_fm_given_D = likelihood(
                    model(self.test_x_final, torch.cat((self.test_i_M, test_i_l)))
                )

                var_p_fl_fm = p_fl_fm_given_D.covariance_matrix

                ## The ellipsis are in case there is a batch dimension at the zeroth dimension
                sigma_square_mM = var_p_fl_fm[
                    ...,
                    self.n ** self.dimension : 2 * self.n ** self.dimension,
                    : self.n ** self.dimension,
                ]
                sigma_square_Mm = var_p_fl_fm[
                    ...,
                    : self.n ** self.dimension,
                    self.n ** self.dimension : 2 * self.n ** self.dimension,
                ]
                sigma_square_m = var_p_fl_fm[
                    ...,
                    self.n ** self.dimension : 2 * self.n ** self.dimension,
                    self.n ** self.dimension : 2 * self.n ** self.dimension,
                ]

                ## In case we have batches
                if len(sigma_square_m.shape) > 2:
                    var_m = torch.zeros(
                        sigma_square_m.shape[0], sigma_square_m.shape[1]
                    )
                    for i in range(sigma_square_m.shape[0]):
                        var_m[i] = torch.diag(sigma_square_m[i])
                else:
                    var_m = torch.diag(sigma_square_m)

                var_p_fl_given_fm_D = torch.zeros_like(var_m)

                if len(sigma_square_m.shape) > 2:
                    for i in range(sigma_square_m.shape[0]):
                        rhs = torch.mm(
                            torch.mm(sigma_square_mM[i], inv_sigma_square_M[i]),
                            sigma_square_Mm[i],
                        )
                        var_p_fl_given_fm_D[i] = var_m[i] - rhs.diag()
                else:
                    rhs = torch.mm(
                        torch.mm(sigma_square_mM, inv_sigma_square_M), sigma_square_Mm
                    )
                    var_p_fl_given_fm_D = var_m - rhs.diag()

                entropy_1 = torch.log(
                    torch.sqrt(2 * math.pi * math.e * p_fl_given_D.variance)
                )
                entropy_2 = torch.log(
                    torch.sqrt(2 * math.pi * math.e * var_p_fl_given_fm_D)
                )

                information = entropy_1 - entropy_2

                ## If batch information (MCMC), then information should be mean of all samples
                if len(information.shape) > 0:
                    information = information.mean(dim=0)

                max_inf, index_max_inf = torch.max(information / lbd, 0)

                print("Maximum Information at", l, ":", max_inf)

                if maximum_information is None or max_inf > maximum_information:
                    maximum_information = max_inf
                    index_maximum_information = index_max_inf
                    it = l

        print("Fidelity chosen:", it)
        return self.test_x[index_max_inf].unsqueeze(0), it

