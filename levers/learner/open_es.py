from copy import deepcopy
from typing import Iterator, List
from numpy import zeros_like

import torch
import torch.optim as optim
from torch.autograd import Variable


class OpenES:

    def __init__(
        self, 
        pop_size: int = 20,
        sigma_init: float = 0.04,
        sigma_decay: float = 0.999,
        sigma_limit: float = 0.01,
        optim_lr: float = 0.01,
        optim_maximize: bool = True
    ):
        """
        OpenAI-ES Salimans et al. (2017))
        Reference: https://arxiv.org/pdf/1703.03864.pdf
        Inspired by: https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/open_es.py
        """
        assert pop_size % 2 == 0
        self.pop_size = pop_size

        # Decay of the noises variance
        self.sigma_init = sigma_init
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit

        # Current level of variance
        self.sigma = sigma_init

        # Optimizer
        self.lr = optim_lr
        self.maximize = optim_maximize

        # Current parameters / population of parameters / population of noise
        self.mean = None
        self.population = None
        self.noise_population = None

    def reset(self, params: Iterator):
        """
        Reset strategy to train parameters similar to `params`.
        """
        # TODO: Should the parameters be copied or initialized to zero?
        # Reset current value of sigma
        self.sigma = self.sigma_init
        # Define parameters to train using ES
        self.mean = []
        for param in params:
            self.mean.append(Variable(param.clone().detach()))
        # Initialize corresponding optimizer
        self.optim = optim.Adam(self.mean, lr=self.lr)

    def ask(self) -> List:
        """
        Returns list of proposed parameter lists.
        """
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
                noise_pos.append(+ noise)
                noise_neg.append(- noise)
                # TODO: Should I add sigma here?
            self.population.append(x_pos)
            self.population.append(x_neg)
            self.noise_population.append(noise_pos)
            self.noise_population.append(noise_neg)
        return self.population

    def tell(self, population_fitness):
        """
        Updates and returns internal mean parameters based on population 
        fitness evaluation externally using Open-ES algorithm. 
        """
        # TODO: Optimize and simplify this loop
        n_params = len(self.noise_population[0])
        grad = []
        for i in range(n_params):
            grad.append(
                # OpenES update. Reference:
                # https://arxiv.org/pdf/1703.03864.pdf
                torch.sum(
                    torch.stack([population_fitness[j] * noise[i] for j, noise in enumerate(self.noise_population)]), 
                    dim=0
                ) / (self.pop_size * self.sigma)
            )

        # Set gradients manually and take gradient step using internal optimizer
        self.optim.zero_grad()
        for i, param in enumerate(self.mean):
            param.grad = -grad[i] if self.maximize else grad[i]
        self.optim.step()

        # Update sigma
        self.sigma = max(self.sigma * self.sigma_decay, self.sigma_limit)

        return self.mean