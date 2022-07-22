from .lever_partner import LeverPartner

from typing import List

import torch


class FixedPatternPartner(LeverPartner):
    """
    Partner agent in iterated lever game playing fixed pattern provided at
    initialization.
    """

    def __init__(self, pattern: List[int]):
        self.pattern = pattern
        self.index = 0
        
    def act(
        self,
        payoffs: torch.Tensor, 
        episode_step: int,
        last_player_action: int,
        last_partner_action: int,
    ) -> int:
        """Returns next element in pattern played by agent. """
        action = self.pattern[self.index]
        self.index = (self.index + 1) % len(self.pattern)
        return action

    def reset(self):
        """Resets the partner agent to restart pattern. """
        self.index = 0