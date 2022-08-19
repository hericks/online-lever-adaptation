import os
import random

from datetime import datetime
from itertools import combinations

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from levers.helpers import generate_binary_patterns, train_drqn_agent
from levers.partners import FixedPatternPartner
from levers.learner import DRQNAgent, DRQNetwork
from levers import IteratedLeverEnvironment


# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Run experiment separately for the following targets
# target_slice = slice(0, 35, 1)
target_slice = slice(35, 70, 1)

# Environment settings
payoffs=[1., 1.]
n_iterations = 100
bootstrap_last_step = True

# Training settings
n_train_evals = 10
n_episodes = 2000
epsilon = 0.3

# DRQN settings
n_hidden_units = 4
capacity = 32
batch_size = 16
lr = 0.005
gamma = 0.975
len_update_cycle = 1
tau = 1e-2

# Output settings
models_path = '../models'
train_stats_path = '../train_stats'
figs_path = '../figures'

experiment_name = 'all_length3'
model_name_template = 'DRQN-{partner_patterns}-{train_id}'

# Setup dummy environment
dummy_env = IteratedLeverEnvironment(
    payoffs=payoffs,
    n_iterations=n_iterations,
    partner=FixedPatternPartner((0, 0, 0)),
    include_step=False,
    include_payoffs=False,
)

patterns = generate_binary_patterns(3)
training_configurations = list(combinations(patterns, 4))
for partner_patterns in training_configurations[target_slice]:
    # Setup training environments
    train_envs = [
        IteratedLeverEnvironment(
            payoffs=payoffs,
            n_iterations=n_iterations+int(bootstrap_last_step),
            partner=FixedPatternPartner(pattern),
            include_step=False,
            include_payoffs=False,
        )
        for pattern in partner_patterns
    ]

    aggregated_train_stats = []
    for train_id in range(n_train_evals):
        print(f"{datetime.now()} {partner_patterns} {train_id:02d}")

        # Setup DRQN agent
        learner = DRQNAgent(
            q_net=DRQNetwork(
                rnn = nn.LSTM(
                    input_size=len(dummy_env.dummy_obs()),
                    hidden_size=n_hidden_units,
                    batch_first=True,
                ),
                fnn = nn.Sequential(
                    nn.Linear(
                        in_features=n_hidden_units,
                        out_features=n_hidden_units),
                    nn.ReLU(),
                    nn.Linear(
                        in_features=n_hidden_units, 
                        out_features=dummy_env.n_actions())
                )
            ),
            capacity=capacity,
            batch_size=batch_size,
            lr=lr,
            gamma=gamma,
            len_update_cycle=len_update_cycle,
            tau=tau,
        )

        # Train DRQN agent
        train_stats = train_drqn_agent(
            agent=learner,
            envs=train_envs,
            n_episodes=n_episodes,
            epsilon=epsilon,
            bootstrap_last_step=bootstrap_last_step
        )
        aggregated_train_stats.append(train_stats)

        # Save model
        model_name = model_name_template.format(
            partner_patterns=partner_patterns,
            train_id=train_id
        )
        out_path = os.path.join(
            models_path, experiment_name, model_name + '.pt')
        torch.save(learner.q_net.state_dict(), out_path)

        # Save train stats
        out_path = os.path.join(
            train_stats_path, experiment_name, model_name + '.pickle')
        torch.save(train_stats, out_path)

        # Save loss curve
        fig_path = os.path.join(
            figs_path, experiment_name, model_name + '-loss-curve.png'
        )
        plt.plot(train_stats['episode'], train_stats['loss'])
        plt.semilogy()
        plt.xlabel('Episode')
        plt.ylabel('DRQN train loss')
        plt.savefig(fig_path)
        plt.close() 
