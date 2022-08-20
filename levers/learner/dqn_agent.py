from copy import deepcopy

import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from .replay_memory import Transition, ReplayMemory
from .utils import polyak_update


# TODO: Should the optimizer be passed as a parameter?
# TODO: Fix Redundancy with done parameter in train() method.


class DQNAgent:
    def __init__(
        self,
        q_net: nn.Module,
        capacity: int,
        batch_size: int,
        lr: float,
        tau: float = 1.0,
        gamma: float = 1.0,
        len_update_cycle: int = 10,
    ):
        """
        Learner suitable for simple Q-Learning with neural network function
        approximation.

        Parameters:
            q_net (nn.Module):
                network to use for q- and target-network
            capacity (int):
                capacity of the replay memory
            batch_size (int):
                batch size to sample from the replay memory
            lr (float):
                learning rate
            tau (float):
                coefficient used for (soft) polyak update
                (tau = 0 -> no update; tau = 1 -> a hard update)
            gamma (float):
                discount factor
            len_update_cycle (int):
                the number of training steps between two target updates
                (= 0,1 -> back-to-back updates)
        """
        # Set q- and target-network (equal at first)
        self.q_net = q_net
        self.target_net = deepcopy(q_net)

        # Parameters specifying the target network updates
        self.tau = tau
        self.n_steps_since_update = 0
        self.len_update_cycle = len_update_cycle

        # Initialize the replay memory and batch size used for updates
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size

        # Initialize optimizer for Q-network
        # (the learned parameters are frequently copied to the target network)
        self.gamma = gamma
        self.lr = lr
        self.optim = optim.RMSprop(self.q_net.parameters(), lr=lr)

    def reset(self, state_dict: Optional[Dict] = None):
        """
        Resets the agent (memory, target updates, optimizer).
        If `params` is non-empty, the passed parameters are cloned into the
        policy network.
        """
        # Reset experience replay
        self.memory = ReplayMemory(self.memory.memory.maxlen)

        # If desired, reset policy network
        if state_dict is not None:
            self.q_net.load_state_dict(state_dict)

        # Reset optimizer
        self.optim = optim.RMSprop(self.q_net.parameters(), lr=self.lr)

        # Reset target network
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.n_steps_since_update = 0

    def act(self, obs: torch.Tensor, epsilon: float = 0) -> Tuple[int, bool]:
        """
        Returns action chosen using epsilon-greedy strategy and boolean flag
        indicating whether the action was chosen greedily (True for greedy
        action, False for random action). If epsilon is set to zero, the action
        is chosen greedily w.r.t. the current q-network.
        """
        # TODO: If the number of actions was stored once, the evaluation of
        # the q-network could be avoided for epsilon-greedy policies.
        q_vals = self.q_net(obs)
        if random.uniform(0, 1) < epsilon:
            return random.randrange(0, len(q_vals)), False
        else:
            return torch.argmax(q_vals).item(), True

    def update_memory(self, transition: Transition):
        """Updates the learners experience."""
        self.memory.push(transition)

    def train(self) -> float:
        """
        Performs a single training step if the agent's experience replay is
        sufficiently filled.

        Returns the current training loss if a training step was taken, -1
        otherwise.
        """
        # Perform training step only if experience replay is sufficiently filled
        if len(self.memory) < self.batch_size:
            return -1.0

        # Obtain batch of samples and convert to stacked tensors
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        states = torch.stack(batch.state)
        actions = torch.tensor(batch.action).view((self.batch_size, 1))
        next_states = torch.stack(batch.next_state)
        rewards = torch.tensor(batch.reward).view_as(actions)
        dones = torch.tensor(batch.done)

        # Compute actual state action values
        state_action_vals = self.q_net(states).gather(1, actions)

        # Compute expected state action values
        next_state_vals = torch.zeros_like(state_action_vals)
        next_state_vals[~dones, :] = (
            self.target_net(next_states[~dones, :])
            .max(1, keepdim=True)[0]
            .detach()
        )
        expected_state_action_vals = self.gamma * next_state_vals + rewards

        # Compute loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_vals, expected_state_action_vals)

        # Optimize model
        # TODO: Should the gradient be clamped?
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # Update target network if necessary
        self.n_steps_since_update += 1
        if self.n_steps_since_update >= self.len_update_cycle:
            polyak_update(
                params=self.q_net.parameters(),
                target_params=self.target_net.parameters(),
                tau=self.tau,
            )
            self.n_steps_since_update = 0

        return loss.item()
