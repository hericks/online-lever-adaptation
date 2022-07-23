from copy import deepcopy
from numpy import zeros_like

import torch


class OpenES:

    def __init__(self, pop_size: int = 20):
        assert pop_size % 2 == 0
        self.pop_size = pop_size

        self.mean = None

    def reset(self, params):
        self.mean = []
        for param in params:
            self.mean.append(param.clone().detach())

    def ask(self):
        # Antithetic sampling of noise
        population = []
        for _ in range(self.pop_size // 2):
            x_pos = []
            x_neg = []
            for param in self.mean:
                noise = torch.rand_like(param)
                x_pos.append(param + noise)
                x_neg.append(param - noise)
            population.append(x_pos)
            population.append(x_neg)
        return population

# class OpenES:

#     def __init__(
#         self,
#         pop_size: int = 20,
#         sigma_init: float = 0.04,
#         sigma_decay: float = 0.999,
#         sigma_limit: float = 0.01,
#         init_min: float = 0.0,
#         init_max: float = 0.0,
#         clip_min: float = torch.finfo.min,
#         clip_max: float = torch.finfo.max
#     ):
#         self.pop_size = pop_size

#         self.sigma_init = sigma_init
#         self.sigma_decay = sigma_decay
#         self.sigma_limit = sigma_limit
#         self.init_min = init_min
#         self.init_max = init_max
#         self.clip_min = clip_min
#         self.clip_max = clip_max

#         self.sigma = 0
#         self.shapes = None
#         self.mean = None

#     def reset(self, params):
#         self.sigma = self.sigma_init
#         self.shapes = []
#         self.mean = []
#         for param in params:
#             self.mean.append(torch.zeros_like(param))
#             self.shapes.append(param.shape)

#     def ask_strategy(self):
#         new_population = []
#         for i in range(self.pop_size // 2):
#             pos = []
#             neg = []
#             for i, shape in enumerate(self.shapes):
#                 z = torch.randn(*shape)
#                 pos.append(self.mean[i] + self.sigma * z)
#                 neg.append(self.mean[i] - self.sigma * z)
#             new_population.append(pos)
#             new_population.append(neg)
#         return new_population

#     def tell_strategy(self):
#         pass