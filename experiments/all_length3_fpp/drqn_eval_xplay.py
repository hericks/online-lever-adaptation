from copy import deepcopy
from itertools import combinations

import numpy as np
import random
import os

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from levers.environment import IteratedLeverEnvironment
from levers.helpers import generate_binary_patterns
from levers.learner import DRQNAgent, DRQNetwork
from levers.evaluators import eval_DRQNAgent_xplay
from levers.helpers import visualize_xplay_matrix

from drqn import get_parser


if __name__ == "__main__":
    # Settings
    seed = 0
    n_agents = 20
    # n_agents = 70

    out_path = "results/crossplay/"
    name_templae = "drqn-crossplay-{n_agents}x{n_agents}-{seed}.png"
    # name_templae = "drqn-complete-crossplay-{n_agents}x{n_agents}.png"

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
            "drqn.conf",
        ),
    ]
    p = get_parser(default_config_files)
    opt = p.parse_args()

    # Sample from pool of DRQN agents
    patterns = generate_binary_patterns(3)
    xplay_patterns = random.sample(list(combinations(patterns, 4)), n_agents)
    # xplay_patterns = list(combinations(patterns, 4))
    try:
        xplay_state_dicts = [
            torch.load(f"models/drqn/DRQN-{pattern}-0.pt")
            for pattern in xplay_patterns
        ]
    except:
        raise ("Failed to load models for xplay agents.")

    # Evaluate crossplay scores
    env = IteratedLeverEnvironment(
        payoffs=opt.payoffs,
        n_iterations=opt.n_iterations + 1,
        include_step=False,
        include_payoffs=False,
    )

    agent1 = DRQNAgent(
        q_net=DRQNetwork(
            rnn=nn.LSTM(
                input_size=len(env.dummy_obs()),
                hidden_size=opt.rnn_hidden_dim,
                batch_first=True,
            ),
            fnn=nn.Sequential(
                nn.Linear(
                    in_features=opt.rnn_hidden_dim,
                    out_features=opt.fnn_hidden_dim,
                ),
                nn.ReLU(),
                nn.Linear(
                    in_features=opt.fnn_hidden_dim,
                    out_features=env.n_actions(),
                ),
            ),
        ),
        capacity=opt.capacity,
        batch_size=opt.batch_size,
        lr=opt.lr,
        tau=opt.tau,
        gamma=opt.gamma,
        len_update_cycle=opt.len_update_cycle,
    )
    agent2 = deepcopy(agent1)

    xplay_results = np.zeros((n_agents, n_agents))
    for id1, state_dict1 in enumerate(xplay_state_dicts):
        for id2, state_dict2 in enumerate(xplay_state_dicts):
            agent1.q_net.load_state_dict(state_dict1)
            agent2.q_net.load_state_dict(state_dict2)
            xplay_results[id1, id2] = eval_DRQNAgent_xplay(
                env, agent1, agent2, True
            )

    visualize_xplay_matrix(
        xplay_results,
        os.path.join(
            out_path, name_templae.format(n_agents=n_agents, seed=seed)
        ),
        vmin=0,
        vmax=100,
    )
