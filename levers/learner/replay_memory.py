from typing import NamedTuple, Union, List
from collections import deque

import random
import torch


class Transition(NamedTuple):
    state: torch.Tensor
    action: int
    next_state: torch.Tensor
    reward: float
    done: bool


class Trajectory(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class ReplayMemory():

    def __init__(self, capacity: int):
        """
        Replay memory to store either state-action-state transitions or 
        trajectories used for q-learning.
        """
        self.memory = deque([], maxlen=capacity)

    def push(self, element: Union[Transition, Trajectory]):
        """Pushes transition to the replay memory. """
        self.memory.append(element)

    def sample(self, batch_size: int) -> List[Union[Transition, Trajectory]]:
        """Returns sample (of size batch_size) of replay memory. """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)