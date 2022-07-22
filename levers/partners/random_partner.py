from .lever_partner import LeverPartner

import torch
import random


class RandomPartner(LeverPartner):
    """Simple random partner agent for the iterated lever game. """

    def act(
        self,
        payoffs: torch.Tensor, 
        episode_step: int,
        last_player_action: int,
        last_partner_action: int,
    ) -> int:
        """Return action choosen uniformly at random. """
        return random.randrange(0, len(payoffs))

    def reset(self):
        """Reset does nothing for a random partner agent. """
        pass