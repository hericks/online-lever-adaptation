from typing import Iterator, List, Dict

import torch
import torch.optim as optim
from torch.autograd import Variable


# TODO: Should parameters be copied or initialized to zero at reset()?


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
        self.means_dict = None
        self.population = None
        self.noise_population = None

    def reset(self, params_dict: Dict[str, Iterator]):
        """
        Reset strategy to train parameters similar to those stored in the
        values of `parametersets`.
        """
        # Reset current value of sigma
        self.sigma = self.sigma_init

        # Define parameters to train using ES
        self.means_dict = dict()
        for params_name, params in params_dict.items():
            self.means_dict[params_name] = []
            for param in params:
                self.means_dict[params_name].append(
                    Variable(param.clone().detach()))

        # Create joint set of variables to optimize and initialize 
        # corresponding optimizer
        vars = [
            param for params in self.means_dict.values() for param in params]
        self.optim = optim.Adam(vars, lr=self.lr)

    def ask(self) -> Dict[str, List[torch.Tensor]]:
        """
        Returns list of proposed parameter lists.
        """
        # Antithetic sampling of noise
        self.population = []
        self.noise_population = []
        for _ in range(self.pop_size // 2):
            x_pos = {}
            x_neg = {}
            noise_pos = {}
            noise_neg = {}
            for params_name, params in self.means_dict.items():
                x_pos[params_name] = []
                x_neg[params_name] = []
                noise_pos[params_name] = []
                noise_neg[params_name] = []
                for param in params:
                    noise = torch.randn_like(param)
                    x_pos[params_name].append(param + self.sigma * noise)
                    x_neg[params_name].append(param - self.sigma * noise)
                    noise_pos[params_name].append(+ noise)
                    noise_neg[params_name].append(- noise)
            self.population.append(x_pos) 
            self.population.append(x_neg) 
            self.noise_population.append(noise_pos)
            self.noise_population.append(noise_neg)
        return self.population

    def tell(
        self, population_fitness: List[float]) -> Dict[str, List[torch.Tensor]]:
        """
        Updates and returns internal dictionary of mean parameters
        based on external population fitness evaluation using Open-ES algorithm. 
        """
        # TODO: Optimize and simplify this loop
        grads_dict = dict()
        for params_name, params in self.means_dict.items():
            grads = []
            n_params = len(params)
            for i in range(n_params):
                grads.append(
                    # OpenES update. Reference:
                    # https://arxiv.org/pdf/1703.03864.pdf
                    torch.sum(
                        torch.stack([
                            population_fitness[j] * noise[params_name][i]
                            for j, noise in enumerate(self.noise_population)
                        ]), dim=0
                    ) / (self.pop_size * self.sigma)
                )
            grads_dict[params_name] = grads

        # Set gradients manually and take gradient step using internal optimizer
        self.optim.zero_grad()
        for params_name, params in self.means_dict.items():
            for i, param in enumerate(params):
                if self.maximize:
                    param.grad = -grads_dict[params_name][i]
                else:
                    grads_dict[params_name][i]
        self.optim.step()

        # Update sigma
        self.sigma = max(self.sigma * self.sigma_decay, self.sigma_limit)

        return self.means_dict