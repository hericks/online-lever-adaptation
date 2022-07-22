import torch


class LeverPartner:
    """Abstract class for partner agent in iterated lever game. """

    def __init__(self):
        """Abstract __init__() method. """
        pass

    def act(
        self,
        payoffs: torch.Tensor, 
        episode_step: int,
        last_player_action: int,
        last_partner_action: int,
    ) -> int:
        """Abstract act() method. """
        raise NotImplementedError

    def reset(self):
        """Abstract reset() method. """
        raise NotImplementedError