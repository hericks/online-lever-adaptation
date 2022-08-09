from typing import List, Optional, Tuple, Union
from xml.dom import NotFoundErr

import torch
from torch import tensor

from .partners import LeverPartner


class IteratedLeverEnvironment:

    def __init__(
        self, 
        payoffs: List[float],
        n_iterations: int, 
        partner: Optional[LeverPartner] = None,
        include_step: bool = True,
        include_payoffs: bool = True
    ):
        """Tensor implementation of the iterated lever environment. """
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
        self.last_player_action = None
        self.last_partner_action = None
        if self.partner:
            self.partner.reset()
        # Return (initial) state for player(s)
        return self._get_obs()

    def step(
        self, action: Union[int, List[int]]
    ) -> Tuple[torch.Tensor, float, bool]:
        """
        Takes a single environment step based on player's action and returns
        the a tuple (new_observation: torch.Tensor, reward: float, done: bool).
        """
        if self.partner:
            partner_action = self.partner.act(
                payoffs=self.payoffs,
                episode_step=self.episode_step,
                last_player_action=self.last_player_action,
                last_partner_action=self.last_partner_action
            )
        else:
            action, partner_action = action[0], action[1]

        # Compute reward
        reward = self.payoffs[action].item() if partner_action == action else 0.

        # Update internal environment state
        self.last_player_action = action
        self.last_partner_action = partner_action
        self.episode_step +=1

        # Tuple[next_obs: torch.Tensor, reward: float, done: bool]
        return (self._get_obs(), reward, self._is_done())  

    def dummy_obs(self) -> torch.Tensor:
        """Returns a dummy observation for shape inference. """
        return self._get_obs()

    def n_actions(self) -> int:
        """Returns the number of possible actions in the environment. """
        return len(self.payoffs)

    def _get_obs(self) -> torch.Tensor:
        """Return the players observation of the current state. """
        empty = tensor([])
        step = tensor([self.episode_step]) if self.include_step else empty
        payoffs = self.payoffs if self.include_payoffs else empty

        player_action_flag = torch.tensor([0] * len(self.payoffs))
        partner_action_flag = torch.tensor([0] * len(self.payoffs))

        # Only filter None (not zero)
        if self.last_player_action is not None:
            player_action_flag[self.last_player_action] = 1
        if self.last_partner_action is not None:            
            partner_action_flag[self.last_partner_action] = 1

        # [current_step? payoffs? partner-action (1-hot)]
        player_obs =torch.cat([step, payoffs, partner_action_flag]) 
        partner_obs = torch.cat([step, payoffs, player_action_flag]) 

        # Discriminate between single and multi agent environment setup
        if self.partner:
            return player_obs
        else:
            return torch.stack(player_obs, partner_obs)

    def _is_done(self) -> bool:
        """Returns whether the environment is in a terminal state."""
        return self.episode_step == self.episode_length