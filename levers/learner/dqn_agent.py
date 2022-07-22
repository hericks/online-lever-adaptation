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
        len_update_cycle: int = 10
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
            update_target_every (int):
                the number of rounds between two target updates
                (=0,1 -> back-to-back updates)
        """
        # Set q- and target-network (equal at first)
        self.q_net = q_net
        self.target_net = deepcopy(q_net)

        # Save how frequently the target network should be updated
        self.n_rounds_since_update = 0
        self.len_update_cycle = len_update_cycle

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

    def update_memory(self, transition: Transition):
        """Updates the learners experience. """
        self.memory.push(transition)

    def train(self, done: bool):
        """
        Pushes transition to the experience replay and performs a single 
        training step. The parameter indicating a terminal state is used to
        termine when to update the learner's target network.
        """
        # Perform training step only if experience replay is sufficiently filled
        if len(self.memory) < self.batch_size:
            return

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
        next_state_vals[~dones,:] = self.target_net(next_states[~dones,:]).max(
            1, keepdim=True)[0].detach()
        expected_state_action_vals = self.gamma * next_state_vals + rewards

        # Compute loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_vals, expected_state_action_vals)

        # Optimize model
        self.optim.zero_grad()
        loss.backward()
        # TODO: Should the gradient be clamped?
        self.optim.step()

        # Update the target model if necessary
        if done:
            self.n_rounds_since_update += 1

        if self.n_rounds_since_update >= self.len_update_cycle:
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.n_rounds_since_update = 0