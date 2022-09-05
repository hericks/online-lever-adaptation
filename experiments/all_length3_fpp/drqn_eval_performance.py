from itertools import combinations

import os

import numpy as np
import torch
import torch.nn as nn

from levers.environment import IteratedLeverEnvironment
from levers.helpers import generate_binary_patterns
from levers.learner import DRQNAgent, DRQNetwork

from drqn import get_parser
from levers.partners.fixed_pattern_agent import FixedPatternPartner


if __name__ == "__main__":
    n_train_evals = 25
    n_evals = 10

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

    patterns = generate_binary_patterns(3)

    # Sample from pool of DRQN agents
    patterns = generate_binary_patterns(3)
    drqn_state_dicts = dict()
    try:
        for train_patterns in combinations(patterns, 4):
            print(f"Reading in models for: {train_patterns}")
            drqn_state_dicts[train_patterns] = [
                torch.load(f"models/drqn/DRQN-{train_patterns}-{train_id}.pt")
                for train_id in range(n_train_evals)
            ]
    except:
        raise ("Failed to load models.")

    results = {
        "return": np.zeros((n_evals, n_train_evals, 8, 70)),
    }
    for tps_id, train_patterns in enumerate(combinations(patterns, 4)):
        print(tps_id)
        for tid in range(n_train_evals):
            for ep_id, eval_pattern in enumerate(patterns):
                for eid in range(n_evals):
                    env = IteratedLeverEnvironment(
                        payoffs=opt.payoffs,
                        n_iterations=opt.n_iterations + 1,
                        partner=FixedPatternPartner(eval_pattern),
                        include_step=False,
                        include_payoffs=False,
                    )

                    agent = DRQNAgent(
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
                    agent.q_net.load_state_dict(
                        drqn_state_dicts[train_patterns][tid]
                    )

                    obs = env.reset()
                    episode_return = 0
                    for step in range(env.episode_length - 1):
                        action = agent.act(obs)
                        next_obs, reward, done = env.step(action)
                        obs = next_obs
                        # Update episode stats
                        episode_return += reward

                    results["return"][eid, tid, ep_id, tps_id] = episode_return

    torch.save(results, "results/drqn-results.pickle")
