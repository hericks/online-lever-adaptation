from copy import deepcopy
from numpy import zeros_like

import torch
import torch.optim as optim
from torch.autograd import Variable


class OpenES:

    def __init__(self, pop_size: int = 20):
        assert pop_size % 2 == 0
        self.pop_size = pop_size
        self.sigma = 1

        #
        self.mean = None
        self.population = None
        self.noise_population = None

        #
        self.maximize = True

    def reset(self, params):
        self.mean = []
        for param in params:
            self.mean.append(Variable(param.clone().detach()))
        # TODO: Specify learning rate
        self.optim = optim.Adam(self.mean, lr=1)

    def ask(self):
        # Antithetic sampling of noise
        self.population = []
        self.noise_population = []
        for _ in range(self.pop_size // 2):
            x_pos = []
            x_neg = []
            noise_pos = []
            noise_neg = []
            for param in self.mean:
                noise = torch.randn_like(param)
                x_pos.append(param + self.sigma * noise)
                x_neg.append(param - self.sigma * noise)
                noise_pos.append(+noise)
                noise_neg.append(-noise)
                # TODO: Should I add sigma here?
            self.population.append(x_pos)
            self.population.append(x_neg)
            self.noise_population.append(noise_pos)
            self.noise_population.append(noise_neg)
        return self.population

    def tell(self, population_fitness):
        n_params = len(self.noise_population[0])
        grad = []
        for i in range(n_params):
            grad.append(
                torch.sum(
                    torch.stack([population_fitness[j] * noise[i] for j, noise in enumerate(self.noise_population)]), 
                    dim=0
                ) / (self.pop_size * self.sigma)
            )
        # Set gradients manually
        self.optim.zero_grad()
        for i, param in enumerate(self.mean):
            param.grad = -grad[i] if self.maximize else grad[i]
        self.optim.step()

        return self.mean