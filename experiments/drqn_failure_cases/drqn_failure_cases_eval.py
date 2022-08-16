# Relative imports outside of package
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

import torch
import torch.nn as nn

from levers import IteratedLeverEnvironment
from levers.partners import FixedPatternPartner
from levers.learner import DRQNAgent
from levers.helpers import eval_drqn_agents, generate_binary_patterns

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

n_train_evals = 10

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
eval_patterns = [(0, 0, 0)]
eval_envs = [
    IteratedLeverEnvironment(
        payoffs, truncated_length+1, FixedPatternPartner(list(pattern)),
        include_step, include_payoffs)
    for pattern in eval_patterns
]

total_res = {}
for train_patterns_id, train_patterns in enumerate(list(itertools.combinations(patterns, 4))):
    print(f'TRAIN-PATTERNS ({train_patterns_id+1:2d}/70): {train_patterns}')

    # Initialize and load DRQN agents
    agents = [
        DRQNAgent(
            DRQNetwork(
                input_size=len(eval_envs[0].dummy_obs()),
                hidden_size=hidden_size,
                n_actions=eval_envs[0].n_actions()
            ),
            capacity, batch_size, lr, gamma, len_update_cycle, tau
        )
        for agent_id in range(n_train_evals)
    ]

    # Load agents' q networks 
    for agent_id in range(n_train_evals):
        in_path = os.path.join(models_dir, model_name_template.format(
            train_patterns=train_patterns,
            eval_id=agent_id,
        ))
        agents[agent_id].q_net =torch.load(in_path)

    # Evaluate agent
    res = eval_drqn_agents(agents, eval_envs, True)
    total_res[train_patterns] = res['rewards'].sum(dim=(0, 2)).item()

total_res_view = [(value, sum([sum(pattern) for pattern in train_pattern]), train_pattern) for train_pattern, value in total_res.items()]
total_res_view.sort(reverse=True)

print('\n\n' + '-' * 100)
for ret, n_ones, train_pattern in total_res_view:
    print(f'{train_pattern} : ({n_ones:2d}) : {ret:5.2f}')

import matplotlib.pyplot as plt

plt.subplot(1, 1, 1)
plt.scatter([n_ones for _, n_ones, _ in total_res_view], [ret for ret, _, _ in total_res_view])
plt.savefig('my_test.png')
plt.close()