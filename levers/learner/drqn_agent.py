from math import nan
from typing import Dict, List, NamedTuple, Tuple
from copy import deepcopy
from dataclasses import dataclass

import random

import torch
import torch.nn as nn
import torch.optim as optim

from levers.learner.replay_memory import ReplayMemory, Trajectory


class DRQNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, n_actions):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.linear = nn.Linear(in_features=hidden_size, out_features=n_actions)

    def forward(self, input, hidden=None):
        # input shape: (batch_size, seq_length, input_size)
        # lstm_out shape: (batch_size, seq_length, hidden_size)
        # out shape: (batch_size, seq_length, n_actions)
        lstm_out, lstm_hid = self.lstm(input, hidden)
        return self.linear(lstm_out), lstm_hid


@dataclass
class TrajectoryBuffer:
    observations: List[torch.Tensor]
    actions: List[int]
    rewards: List[float]
    dones: List[bool]


class DRQNAgent():

    def __init__(
        self,
        q_net: nn.Module,
        capacity: int,
        batch_size: int,
        lr: float,
        gamma: float = 1.0,
        len_update_cycle: int = 10,
        tau: float = 1.0,
    ):
        """
        Simple implementation of Learner performung simple DRQ-Learning.
        """
        # Set q- and target-network (equal at first)
        self.q_net = q_net
        self.target_net = deepcopy(q_net)

        self.replay_memory = ReplayMemory(capacity=capacity)
        self.buffer: TrajectoryBuffer = None
        self.batch_size = batch_size

        # Save how frequently the target network should be updated
        self.n_rounds_since_update = 0
        self.len_update_cycle = len_update_cycle
        self.tau = tau

        self.gamma = 1.0
        self.hidden=None

        # Initialize optimizer for DRQ-network
        # (the learned parameters are frequently copied to the target network)
        self.optim = optim.RMSprop(self.q_net.parameters(), lr=lr)

    def reset_trajectory_buffer(self, init_obs: torch.Tensor):
        self.episode_hist = TrajectoryBuffer(
            observations=[init_obs], actions=[], rewards=[], dones=[]
        )
        self.hidden = None

    def update_trajectory_buffer(
        self, 
        action: int,
        reward: float,
        next_obs: torch.Tensor,
        done: bool
    ):  
        # Append experience to current episode buffer
        self.buffer.actions.append(action)
        self.buffer.rewards.append(reward)
        self.buffer.observations.append(next_obs)
        self.buffer.dones.append(done)

        # If experience was collected at the end of an episde, automatically
        # transfer the current episode buffer to the replay memory.
        if done:
            self.flush_trajectory_buffer()

    def flush_trajectory_buffer(self):
        if self.buffer:
            trajectory = Trajectory(
                torch.stack([o for o in self.buffer.observations]),
                torch.stack([torch.tensor([a]) for a in self.buffer.actions]),
                torch.stack([torch.tensor([r]) for r in self.buffer.rewards]),
                torch.stack([torch.tensor([d]) for d in self.buffer.dones]),
            )
            self.replay_memory.push(trajectory)
            self.buffer = None

        # Increase number of rounds since last policy->target update and
        # update the target model if necessary
        # NOTE: When 0 <= self.tau < 1 a soft update is used
        self.n_rounds_since_update += 1
        if self.n_rounds_since_update >= self.len_update_cycle:
            tps = self.target_net.parameters()
            qps = self.q_net.parameters()
            for tp, qp in zip(tps, qps):
                    tp_update = self.tau * qp.data + (1.0 - self.tau) * tp.data
                    tp.data.copy_(tp_update)
            self.n_rounds_since_update = 0

    def act(self, obs: torch.Tensor, epsilon: float = 0) -> int:
        """
        Returns action chosen using epsilon-greedy strategy. If epsilon is set
        to zero, the action is chosen greedily w.r.t. the current q-network.
        """
        # TODO: If the number of actions was stored once, the evaluation of 
        # the q-network could be avoided for epsilon-greedy policies.
        q_vals, self.hidden = self.q_net(obs.unsqueeze(0), self.hidden)
        if random.uniform(0, 1) < epsilon:
            return random.randrange(0, q_vals.shape[1])
        else:
            return torch.argmax(q_vals).item()

    def train(self):
        if len(self.replay_memory) < self.batch_size:
            return None

        # Sample episodes from replay memory
        episodes = self.replay_memory.sample(self.batch_size)

        # Obtain batched obervations, actions, rewards, and done indicators
        obs_batch = torch.stack([e.observations for e in episodes])
        actions_batch = torch.stack([e.actions for e in episodes])
        rewards_batch = torch.stack([e.rewards for e in episodes])
        dones_batch = torch.stack([e.dones for e in episodes])

        # Compute action state values for all observations but the last in
        # the sequence
        q_values, _ = self.q_net(obs_batch)
        state_action_values = q_values[:,:-1,:].gather(2, actions_batch)

        # Compute expected state action values for those states
        next_state_values = torch.zeros_like(state_action_values)
        target_q_values, _ = self.target_net(obs_batch)
        target_next_state_values = target_q_values[:,:-1,:].max(2, True)[0].detach()
        next_state_values[~dones_batch] = target_next_state_values[~dones_batch]
        expected_state_action_values = self.gamma * next_state_values + rewards_batch

        # Compute loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        # Optimize model
        # TODO: Should the gradient be clamped?
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item()