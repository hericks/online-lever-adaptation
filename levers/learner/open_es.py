from typing import Any, Callable, Iterable, Optional, Union

import torch

from evotorch.core import Problem
from evotorch.tools import rowwise_sum as total
from evotorch.tools import multiply_rows_by_scalars as dot
from evotorch.tools import RealOrVector, modify_tensor, Device, DType
from evotorch.algorithms.distributed import GaussianSearchAlgorithm
from evotorch.distributions import SymmetricSeparableGaussian


class SymmetricSeparableGaussianOES(SymmetricSeparableGaussian):
    """Separable Multivariate Gaussian, as used by OpenES."""

    MANDATORY_PARAMETERS = {"mu", "sigma", "sigma_decay", "sigma_min"}

    def _compute_gradients(
        self,
        samples: torch.Tensor,
        weights: torch.Tensor,
        ranking_used: Optional[str]
    ) -> dict:
        """Computes OpenES gradient based on samples and weights. """
        mu = self.mu
        sigma = self.sigma

        # Compute the scaled noises, that is, the noise vectors which
        # were used for generating the solutions
        # (solution = scaled_noise + center)
        scaled_noises = samples[0::2] - mu

        # Separate the plus and the minus ends of the directions
        fdplus = weights[0::2]
        fdminus = weights[1::2]

        # Compute gradient
        [num_solutions] = weights.shape
        num_directions = num_solutions // 2
        grad_mu = total(
            dot(fdplus - fdminus, scaled_noises)) / (sigma * num_directions)

        return {
            "mu": grad_mu,
            "sigma": torch.tensor(0),
        }

    def update_parameters(
        self,
        gradients: dict,
        *,
        learning_rates: Optional[dict] = None,
        optimizers: Optional[dict] = None,
    ) -> "SymmetricSeparableGaussianOES":
        """Performs OpenES parameter update. """
        mu_grad = gradients["mu"]

        # Compute new mean and sigma
        new_mu = self.mu + self._follow_gradient(
            "mu", mu_grad, learning_rates=learning_rates, optimizers=optimizers
        )
        new_sigma = torch.maximum(
            self.sigma * self.parameters['sigma_decay'],
            torch.ones_like(self.sigma) * self.parameters['sigma_min']
        )

        return self.modified_copy(mu=new_mu, sigma=new_sigma)


class OpenES(GaussianSearchAlgorithm):

    DISTRIBUTION_TYPE = NotImplemented  # To be filled by the OpenES instance
    DISTRIBUTION_PARAMS = NotImplemented  # To be filled by the OpenES instance

    def __init__(
        self,
        problem: Problem,
        *,
        popsize: int,
        learning_rate: float,
        stdev_init: float = 0.5,
        stdev_decay: float = 0.99,
        stdev_min: float = 0.1,
        optimizer="adam",
        optimizer_config: Optional[dict] = None,
        mean_init: Optional[RealOrVector] = None,
        obj_index: Optional[int] = None,
        distributed: bool = False,
    ):
        # Define distribution suitable for OpenES parameter updates.
        self.DISTRIBUTION_TYPE = SymmetricSeparableGaussianOES
        self.DISTRIBUTION_PARAMS = {
            "sigma_decay": stdev_decay,
            "sigma_min": stdev_min,
        }

        super().__init__(
            problem,
            popsize=popsize,
            center_learning_rate=learning_rate,
            stdev_learning_rate=0.0,
            stdev_init=stdev_init,
            radius_init=None,
            popsize_max=popsize,
            num_interactions=None,
            optimizer=optimizer,
            optimizer_config=optimizer_config,
            ranking_method=None,
            center_init=mean_init,
            stdev_min=stdev_min,
            stdev_max=stdev_init,
            stdev_max_change=1.,
            obj_index=obj_index,
            distributed=distributed,
            popsize_weighted_grad_avg=None,
            ensure_even_popsize=True,
        )