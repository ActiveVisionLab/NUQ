import torch


class HighFidelityUCB:
    def __init__(self):
        self.nmb_smpl = 0

    def __call__(self, data, model, likelihood):
        self.nmb_smpl += 1  # Number of times we have explored in high fidelity

        test_x = data.x_test
        test_i_l = torch.full(
            (test_x.shape[0],), fill_value=data.number_tasks - 1, dtype=torch.long
        )
        with torch.no_grad():
            f = likelihood(model(test_x, test_i_l))

        mean = f.mean
        var = f.variance

        ### This implements the equation for ECB according to
        #  Srinvas (2010) Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design
        # xt = argmax (mu_{t-1}(x) + sqrt(\beta_t * variance_{t-1}(x))) # Note that it's sqrt variance which is std
        # \beta_t = 2log(|D|t^2*pi^2/6*delta)
        # delta = 0.1
        # D is the decision set
        # Also note that in the paper they scale beta down by a factor of 5, which will be adopted here as well
        delta = 0.1
        beta_t = 2 * torch.log(
            torch.tensor(
                (test_x.size()[0] * self.nmb_smpl ** 2 * 3.1415 ** 2) / (6 * delta)
            )
        )
        beta_t /= 5

        acq_values = mean + torch.sqrt(beta_t * var)

        ## This is only in case we do MCMC on hyperparameters
        if len(acq_values.shape) > 1:
            acq_values = acq_values.mean(axis=0)
        return test_x[torch.argmax(acq_values)].unsqueeze(0)
