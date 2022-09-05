from itertools import combinations
from copy import deepcopy

import numpy as np
import random
import os

import torch
import torch.nn as nn

from levers import IteratedLeverEnvironment
from levers.partners import FixedPatternPartner
from levers.learner import DQNAgent
from levers.helpers import generate_binary_patterns
from levers.evaluators import eval_DQNAgent

from odql import get_parser


if __name__ == "__main__":
    # Settings
    seed = 0

    # Exploration strategies
    schedule1 = lambda step, episode_length: max(
        0.0, 0.4 * (1 - 4 * step / episode_length)
    )
    schedule2 = lambda step, episode_length: max(
        0.0, 0.4 * (1 - 4 * step / episode_length)
    )

    # Make reproducible
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load experiment configuration
    default_config_files = [
        os.path.join(
            "/data/engs-oxfair3/orie4536/online-lever-adaptation/",
            "experiments",
            "all_length3_fpp",
            "defaults.conf",
        ),
        os.path.join(
            "/data/engs-oxfair3/orie4536/online-lever-adaptation/",
            "experiments",
            "all_length3_fpp",
            "odql.conf",
        ),
    ]
    p = get_parser(default_config_files)
    opt = p.parse_args()

    # Sample from pool of DRQN agents
    patterns = generate_binary_patterns(3)

    for tps in combinations(patterns, 4):
        try:
            old_odql_state_dict = torch.load(f"models/odql-old/ODQL-{tps}-0.pt")
            new_odql_state_dict = torch.load(f"models/odql/ODQL-{tps}-0.pt")

            old_hist_state_dict = torch.load(f"models/odql-old/HIST-{tps}-0.pt")
            new_hist_state_dict = torch.load(f"models/odql/HIST-{tps}-0.pt")
        except:
            raise (f"Failed to load models for {tps}.")

        train_envs = [
            IteratedLeverEnvironment(
                payoffs=opt.payoffs,
                n_iterations=opt.n_iterations + 1,
                partner=FixedPatternPartner(train_pattern),
                include_step=False,
                include_payoffs=False,
            )
            for train_pattern in tps
        ]
        test_envs = [
            IteratedLeverEnvironment(
                payoffs=opt.payoffs,
                n_iterations=opt.n_iterations + 1,
                partner=FixedPatternPartner(test_pattern),
                include_step=False,
                include_payoffs=False,
            )
            for test_pattern in patterns
            if test_pattern not in tps
        ]

        hist_rep = nn.LSTM(
            input_size=len(train_envs[0].dummy_obs()),
            hidden_size=opt.hist_rep_dim,
        )

        agent = DQNAgent(
            q_net=nn.Sequential(
                nn.Linear(opt.hist_rep_dim, opt.dqn_hidden_dim),
                nn.ReLU(),
                nn.Linear(opt.dqn_hidden_dim, train_envs[0].n_actions()),
            ),
            capacity=opt.capacity,
            batch_size=opt.batch_size,
            lr=opt.dqn_lr,
            tau=opt.tau,
            len_update_cycle=opt.len_update_cycle,
        )

        # --- TRAINING SCORES
        hist_rep.load_state_dict(old_hist_state_dict)
        agent.q_net.load_state_dict(old_odql_state_dict)
        old_train_scores = eval_DQNAgent(
            learner=agent,
            hist_rep=hist_rep,
            envs=train_envs,
            bootstrap_last_step=True,
            train=True,
            epsilon=opt.epsilon,
        )

        hist_rep.load_state_dict(new_hist_state_dict)
        agent.q_net.load_state_dict(new_odql_state_dict)
        new_train_scores = eval_DQNAgent(
            learner=agent,
            hist_rep=hist_rep,
            envs=train_envs,
            bootstrap_last_step=True,
            train=True,
            epsilon=opt.epsilon,
        )

        old_train = (
            old_train_scores["greedy_return"]
            / old_train_scores["n_greedy_steps"]
        )
        new_train = (
            new_train_scores["greedy_return"]
            / new_train_scores["n_greedy_steps"]
        )

        # --- TESTING SCORES
        hist_rep.load_state_dict(old_hist_state_dict)
        agent.q_net.load_state_dict(old_odql_state_dict)
        old_test_scores = eval_DQNAgent(
            learner=agent,
            hist_rep=hist_rep,
            envs=test_envs,
            bootstrap_last_step=True,
            train=True,
            epsilon=opt.epsilon,
        )

        hist_rep.load_state_dict(new_hist_state_dict)
        agent.q_net.load_state_dict(new_odql_state_dict)
        new_test_scores = eval_DQNAgent(
            learner=agent,
            hist_rep=hist_rep,
            envs=test_envs,
            bootstrap_last_step=True,
            train=True,
            epsilon=opt.epsilon,
        )

        old_test = (
            old_test_scores["greedy_return"] / old_test_scores["n_greedy_steps"]
        )
        new_test = (
            new_test_scores["greedy_return"] / new_test_scores["n_greedy_steps"]
        )

        print(f"{tps}: TRAIN {old_train:3.2f} -> {new_train:3.2f}", end=" | ")
        print(f"TEST {old_test:3.2f} -> {new_test:3.2f}")
