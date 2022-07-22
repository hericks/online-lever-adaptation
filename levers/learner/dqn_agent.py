from copy import deepcopy

import random

import torch
import torch.nn as nn
import torch.optim as optim

from .replay_memory import Transition, ReplayMemory


# TODO: Should the optimizer be passed as a parameter?
# TODO: Fix Redundancy with done parameter in train() method.


class DQNAgent():
    """Learner suitable for simple Q-Learning. """

    def __init__(
        self,
        q_net: nn.Module,
        capacity: int,
        batch_size: int,
        lr: float,
        gamma: float = 1.0,
        n_rounds_between_updates: int = 10
    ):
        """
        Instantiate simple DQN-Agent with the following parameters:
            q_net (nn.Module):
                network to use for q- and target-network
            capacity (int):
                capacity of the replay memory
            batch_size (int):
                batch size to sample from the replay memory
            lr (float):
                learning rate
            gamma (float):
                discount factor
            n_rounds_between_updates (int):
                the number of rounds necessary to play 
        """
        # Set q- and target-network (equal at first)
        self.q_net = q_net
        self.target_net = deepcopy(q_net)

        # Save how frequently the target network should be updated
        self.n_rounds_since_last_target_update = 0
        self.n_rounds_between_update = n_rounds_between_updates

        # Initialize the replay memory and batch size used for updates
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        # Initialize optimizer for Q-network
        # (the learned parameters are frequently copied to the target network)
        self.optim = optim.RMSprop(self.q_net.parameters(), lr=lr)
        self.gamma = gamma

    def act(self, obs: torch.Tensor, epsilon: float = 0) -> int:
        """
        Returns action chosen using epsilon-greedy strategy. If epsilon is set
        to zero, the action is chosen greedily w.r.t. the current q-network.
        """
        # TODO: If the number of actions was stored once, the evaluation of 
        # the q-network could be avoided for epsilon-greedy policies.
        q_vals = self.q_net(torch.zeros_like(obs))
        if random.uniform(0, 1) < epsilon:
            return random.randrange(0, len(q_vals))
        else:
            return torch.argmax(q_vals).item()

    def train(self, transition: Transition, done: bool = False):
        """
        Performs a single training step. If a transition is passed, it is 
        pushed to the experience memory first. 
        """
        if transition:
            self.memory.push(transition)
        pass