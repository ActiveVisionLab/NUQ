import matplotlib.pyplot as plt
import torch


def plot(data, model, likelihood):
    number_tasks = data.number_tasks

    fig, axs = plt.subplots(1, number_tasks)
    test_x = data.x_test

    for i in range(number_tasks):
        test_i_l = torch.full((test_x.shape[0],), fill_value=i, dtype=torch.long)
        with torch.no_grad():
            f = model(test_x, test_i_l)
            lower, upper = f.confidence_region()
        print(test_x.numpy())
        print(lower.numpy())
        print(upper.numpy())
        axs[i].fill_between(
            test_x.squeeze().numpy(), lower.numpy(), upper.numpy(), alpha=0.5
        )
        axs[i].plot(test_x, f.mean.numpy())
        axs[i].plot(data.get_x(i), data.get_y(i), "k*")

    plt.show()


def plot_mcmc(data, model, likelihood, number_means=20):
    test_x = data.x_test

    fig, axs = plt.subplots(1, data.number_tasks)
    for i in range(data.number_tasks):

        expanded_test_x = test_x.unsqueeze(0).repeat(data.num_samples, 1, 1)

        test_i_l = torch.full((test_x.shape[0],), fill_value=i, dtype=torch.long)
        with torch.no_grad():
            f = likelihood(model(test_x, test_i_l))

        for j in range(number_means):
            axs[i].plot(test_x, f.mean[j].numpy())

        axs[i].plot(data.get_x(i), data.get_y(i), "k*")

    plt.show()
