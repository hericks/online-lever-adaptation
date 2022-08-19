from math import nan
from typing import Dict, List, NamedTuple, Tuple
from copy import deepcopy
from dataclasses import dataclass

import random

import torch
import torch.nn as nn
import torch.optim as optim

from levers.learner.replay_memory import (
    ReplayMemory, Trajectory, TrajectoryBuffer
)
from levers.learner.utils import polyak_update

class DRQNetwork(nn.Module):

    def __init__(self, rnn: nn.modules.RNNBase, fnn: nn.Module):
        """
        NOTE: This always sets `batch_first` for the RNN to `True`. 
        """
        super().__init__()
        self.rnn = rnn
        self.rnn.batch_first = True
        self.fnn = fnn

    def forward(self, input, hidden=None):
        # input shape: (batch_size, seq_length, input_size)
        # rnn_out shape: (batch_size, seq_length, hidden_size)
        # out shape: (batch_size, seq_length, n_actions)
        rnn_out, rnn_hid = self.rnn(input, hidden)
        return self.fnn(rnn_out), rnn_hid

# class DRQNetwork(nn.Module):

#     def __init__(self, input_size, hidden_size, n_actions):
#         super().__init__()
#         self.lstm = nn.LSTM(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             batch_first=True,
#         )
#         self.linear = nn.Linear(in_features=hidden_size, out_features=n_actions)

#     def forward(self, input, hidden=None):
#         # input shape: (batch_size, seq_length, input_size)
#         # lstm_out shape: (batch_size, seq_length, hidden_size)
#         # out shape: (batch_size, seq_length, n_actions)
#         lstm_out, lstm_hid = self.lstm(input, hidden)
#         return self.linear(lstm_out), lstm_hid


class DRQNAgent():

    def __init__(
        self,
        q_net: DRQNetwork,
        capacity: int,
        batch_size: int,
        lr: float,
        gamma: float = 1.0,
        len_update_cycle: int = 10,
        tau: float = 1.0,
    ):
        """
        Learner suitable for simple Q-Learning with recurrent neural network
        function approximation for usage in partially observable domains.

        Parameters:
            q_net (DRQNetwork):
                network to use for q- and target-network
            capacity (int):
                capacity of the replay memory (number of episodes stored)
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
        self.n_episodes_since_update = 0
        self.len_update_cycle = len_update_cycle

        # Initialize the replay memory, current trajectory buffer, and
        # batch size used for updates
        self.replay_memory = ReplayMemory(capacity=capacity)
        self.buffer: TrajectoryBuffer = None
        self.batch_size = batch_size

        # Initialize optimizer for recurrent Q-network
        # (the learned parameters are frequently copied to the target network)
        self.gamma = gamma
        self.lr = lr
        self.optim = optim.RMSprop(self.q_net.parameters(), lr=lr)

        # Internal hidden states used for acting
        self.hidden=None

    def reset(self):
        # Reset experience replay
        capacity = self.replay_memory.memory.maxlen
        self.replay_memory = ReplayMemory(capacity)

        # Reset optimizer
        self.optim = optim.RMSprop(self.q_net.parameters(), lr=self.lr)

        # Reset target network
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.n_episodes_since_update = 0

    def reset_new_episode(self, init_obs: torch.Tensor):
        """
        Resets the agent's internal state and fills trajectory buffer with
        initial observation `init_obs`.
        """
        self.hidden = None
        self.buffer = TrajectoryBuffer(
            observations=[init_obs], actions=[], rewards=[], dones=[]
        )

    def update_trajectory_buffer(
        self, 
        action: int,
        reward: float,
        next_obs: torch.Tensor,
        done: bool
    ):  
        """Updates the trajectory buffer with the passed experience passed. """
        # Append experience to trajectory buffer
        self.buffer.actions.append(action)
        self.buffer.rewards.append(reward)
        self.buffer.observations.append(next_obs)
        self.buffer.dones.append(done)

        # If experience was collected at last step of an episode, flush the 
        # trajectory buffer automatically
        if done:
            self.flush_trajectory_buffer()

    def flush_trajectory_buffer(self):
        """
        Marks the end of an episode. Flushes the trajectory buffer to the 
        agent's replay memory.
        """
        if not self.buffer:
            return

        trajectory = Trajectory(
            torch.stack([o for o in self.buffer.observations]),
            torch.stack([torch.tensor([a]) for a in self.buffer.actions]),
            torch.stack([torch.tensor([r]) for r in self.buffer.rewards]),
            torch.stack([torch.tensor([d]) for d in self.buffer.dones]),
        )
        self.replay_memory.push(trajectory)
        self.buffer = None


    def act(self, obs: torch.Tensor, epsilon: float = 0) -> int:
        """
        Returns action chosen using epsilon-greedy strategy and advances
        internal hidden state. If epsilon is set to zero, the action is chosen
        greedily w.r.t. the current q-network.
        """
        q_vals, self.hidden = self.q_net(obs.unsqueeze(0), self.hidden)
        if random.uniform(0, 1) < epsilon:
            return random.randrange(0, q_vals.shape[1])
        else:
            return torch.argmax(q_vals).item()

    def train(self) -> float:
        """
        Performs single training step if the agent's experience replay is
        sufficiently filled.

        Returns the current training loss if a training step was taken, -1
        otherwise.
        """
        if len(self.replay_memory) < self.batch_size:
            return None

        # Sample episodes from replay memory
        episodes = self.replay_memory.sample(self.batch_size)

        # Obtain batched obervations, actions, rewards, and done indicators
        obs = torch.stack([e.observations for e in episodes])
        actions = torch.stack([e.actions for e in episodes])
        rewards = torch.stack([e.rewards for e in episodes])
        dones = torch.stack([e.dones for e in episodes])

        # Compute action state values for all observations but the last in
        # the sequence
        q_values = self.q_net(obs)[0][:,:-1,:]
        state_action_values = q_values.gather(2, actions)

        # Compute expected state action values for those states
        next_state_values = torch.zeros_like(state_action_values)
        target_q_values = self.target_net(obs)[0][:,1:,:]
        target_next_state_values = target_q_values.max(2, True)[0].detach()
        next_state_values[~dones] = target_next_state_values[~dones]
        expected_state_action_values = self.gamma * next_state_values + rewards

        # Compute loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        # Optimize model
        # TODO: Should the gradient be clamped?
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # Update target network if necessary
        self.n_episodes_since_update += 1
        if self.n_episodes_since_update >= self.len_update_cycle:
            polyak_update(
                params=self.q_net.parameters(),
                target_params=self.target_net.parameters(),     
                tau=self.tau,
            )
            self.n_episodes_since_update = 0

        return loss.item()