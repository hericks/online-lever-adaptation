from typing import List, Tuple

import torch
from torch import tensor

from .partners import LeverPartner



class IteratedLeverEnvironment:
    """Tensor implementation of the iterated lever environment. """

    def __init__(
        self, 
        payoffs: List,
        n_iterations: int, 
        partner: LeverPartner,
        include_step: bool = True,
        include_payoffs: bool = True
    ):
        # Tensor of lever payoffs of shape torch.Size([n_levers])
        self.payoffs = torch.tensor(payoffs, dtype=torch.float32)
        self.partner = partner

        # Episode length and number number of steps taken in current episode
        self.episode_length = n_iterations
        self.episode_step = 0

        # Defines how the player's oberservations are constructed
        self.include_step = include_step
        self.include_payoffs = include_payoffs

        # Last actions taken
        self.last_player_action = None
        self.last_partner_action = None

    def reset(self) -> torch.Tensor:
        """Reset the environment (including the partner policy). """
        self.episode_step = 0 
        self.partner.reset()
        self.last_player_action = None
        self.last_partner_action = None
        # Return (initial) state for player
        return self._get_obs()

    def step(self, action) -> Tuple[torch.Tensor, float, bool]:
        """
        Takes a single environment step based on player's action and returns
        the a tuple (new_observation: torch.Tensor, reward: float, done: bool).
        """
        partner_action = self.partner.act(
            payoffs=self.payoffs,
            episode_step=self.episode_step,
            last_player_action=self.last_player_action,
            last_partner_action=self.last_partner_action
        )
        self.episode_step +=1
        done = self.episode_step == self.episode_length
        return (tensor([0, 0, 0]), 1.0, done)  

    def _get_obs(self):
        """Return the players observation of the current state. """
        empty = tensor([])
        step = tensor([self.episode_step]) if self.include_step else empty
        payoffs = self.payoffs if self.include_payoffs else empty
        partner_action = torch.tensor([0] * len(self.payoffs))
        # Only filter None (not zero)
        if self.last_partner_action is not None:            
            partner_action[self.last_partner_action] = 1
        # [current_step? payoffs? partner-action (1-hot)]
        return torch.cat([step, payoffs, partner_action])