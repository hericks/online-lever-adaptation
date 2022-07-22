from .partners import LeverPartner

import torch
from torch import tensor



class IteratedLeverEnvironment:
    """Tensor implementation of the iterated lever environment. """

    def __init__(
        self, 
        payoffs: torch.Tensor,
        n_iterations: int, 
        partner: LeverPartner,
        include_step: bool = True,
        include_payoffs: bool = True
    ):
        # tensor of lever payoffs of shape torch.Size([n_levers])
        self.payoffs = payoffs
        self.partner = partner

        # episode length and number number of steps taken in current episode
        self.n_iterations = n_iterations
        self.episode_step = 0

        # defines how the player's oberservations are constructed
        self.include_step = include_step
        self.include_payoffs = include_payoffs

        # last actions taken
        self.last_action = None
        self.last_partner_action = None

    def reset(self):
        """Reset the environment (including the partner policy). """
        self.episode_step = 0 
        self.partner.reset()
        self.last_action = None
        self.last_partner_action = None
        # return (initial) state for player
        return self._get_obs()

    def _get_obs(self):
        """Return the players observation of the current state. """
        empty = tensor([])
        step = tensor([self.episode_step]) if self.include_step else empty
        payoffs = self.payoffs if self.include_payoffs else empty
        partner_action = torch.tensor([0] * len(self.payoffs))
        # only filter None (not zero)
        if self.last_partner_action is not None:            
            partner_action[self.last_partner_action] = 1
        # [current_step? payoffs? partner-action (1-hot)]
        return torch.cat([step, payoffs, partner_action])