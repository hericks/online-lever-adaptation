# Relative imports outside of package
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

from levers import IteratedLeverEnvironment
from levers.partners import FixedPatternPartner
from levers.learner import DRQNAgent
from levers.helpers import train_drqn_agent, generate_binary_patterns

import itertools
import random
import torch
import torch.nn as nn


# Reproducibility
seed = 42
torch.manual_seed(seed)
random.seed(seed)

# Environment settings
payoffs = [1., 1.]
truncated_length = 100
include_step=False
include_payoffs=False

# Learner settings
hidden_size = 4
capacity = 8
batch_size = 4
lr = 0.01
gamma = 0.99
len_update_cycle = 4
tau = 5e-4

# Training settings
n_episodes = 1000
epsilon = 0.3

# Evaluation settings
n_evaluations = 10

# Model settings
models_dir = 'models'
model_name_template = 'drqn-net-train-patterns={train_patterns}-eval-id={eval_id:02d}.pt'


class DRQNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, n_actions):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.lin1 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(in_features=hidden_size, out_features=n_actions)

    def forward(self, input, hidden=None):
        # input shape: (batch_size, seq_length, input_size)
        # lstm_out shape: (batch_size, seq_length, hidden_size)
        # out shape: (batch_size, seq_length, n_actions)
        lstm_out, lstm_hid = self.lstm(input, hidden)
        temp = self.relu(self.lin1(lstm_out))
        return self.lin2(temp), lstm_hid


# Begin experiment
patterns = generate_binary_patterns(3)
for train_patterns in itertools.combinations(patterns, 4):
    print(f'TRAIN-PATTERNS: {train_patterns}')

    # Construct list of environments to train on
    train_envs = [
        IteratedLeverEnvironment(
            payoffs, truncated_length+1, FixedPatternPartner(list(pattern)),
            include_step, include_payoffs)
        for pattern in train_patterns
    ]

    for eval_id in range(n_evaluations):
        print(f'-> evaluation: {eval_id+1:d}/{n_evaluations}')

        # Initialize new learner
        learner = DRQNAgent(
            DRQNetwork(
                input_size=len(train_envs[0].dummy_obs()),
                hidden_size=hidden_size,
                n_actions=train_envs[0].n_actions()
            ),
            capacity, batch_size, lr, gamma, len_update_cycle, tau
        )

        # Train learner
        train_drqn_agent(
            agent=learner,
            envs=train_envs,
            n_episodes=n_episodes,
            epsilon=0.3,
            bootstrap_last_step=True,
        )

        # Save learner's current q network
        out_path = os.path.join(models_dir, model_name_template.format(
            train_patterns=train_patterns,
            eval_id=eval_id,
        ))
        torch.save(learner.q_net, out_path)