# Relative imports outside of package
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

import torch
import torch.nn as nn

from levers import IteratedLeverEnvironment
from levers.partners import FixedPatternPartner
from levers.learner import DRQNAgent
from levers.helpers import eval_drqn_agent, generate_binary_patterns

import itertools


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

# Evaluation settings
n_evaluations = 2

# Model settings
models_dir = 'experiments/drqn_failure_cases/models'
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

# Construct list of environments to evaluate DRQN agent in
eval_envs = [
    IteratedLeverEnvironment(
        payoffs, truncated_length+1, FixedPatternPartner(list(pattern)),
        include_step, include_payoffs)
    for pattern in patterns
]

for train_patterns_id, train_patterns in enumerate(list(itertools.combinations(patterns, 4))):
    print(f'TRAIN-PATTERNS ({train_patterns_id+1:2d}/70): {train_patterns}')

    # Initialize new agent
    agent = DRQNAgent(
        DRQNetwork(
            input_size=len(eval_envs[0].dummy_obs()),
            hidden_size=hidden_size,
            n_actions=eval_envs[0].n_actions()
        ),
        capacity, batch_size, lr, gamma, len_update_cycle, tau
    )

    for train_id in range(10):
        # Load agent's q network
        in_path = os.path.join(models_dir, model_name_template.format(
            train_patterns=train_patterns,
            eval_id=train_id,
        ))
        agent.q_net =torch.load(in_path)

        # Evaluate agent
        res = eval_drqn_agent(agent, eval_envs, True)
        print(res['rewards'].sum(dim=1))
