import torch


class LeverPartner:

    def __init__(self):
        """Abstract __init__() method. """
        pass

    def act(
        self,
        payoffs: torch.Tensor, 
        episode_step: int,
        last_player_action: int,
        last_partner_action: int,
    ):
        """Abstract act() method. """
        raise NotImplementedError

    def reset(self):
        """Abstract reset() method. """
        raise NotImplementedError