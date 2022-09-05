from copy import deepcopy
from itertools import combinations
from threading import current_thread

import matplotlib.pyplot as plt
import numpy as np
import random
import os

import torch
import torch.nn as nn
from levers.environment import IteratedLeverEnvironment
from levers.evaluators import eval_DQNAgent, eval_DQNAgent_xplay

from levers.helpers import generate_binary_patterns
from levers.learner import DQNAgent
from levers.partners.fixed_pattern_agent import FixedPatternPartner

from odql import get_parser

if __name__ == "__main__":
    # Settings
    seed = 0
    n_train_evals = 5
    n_evals = 5

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

    print(f"ODQL. n_train_evals: {n_train_evals}. n_evals: {n_evals}")

    # Sample from pool of DRQN agents
    patterns = generate_binary_patterns(3)
    odql_state_dicts = dict()
    hist_state_dicts = dict()
    try:
        for train_patterns in combinations(patterns, 4):
            odql_state_dicts[train_patterns] = [
                torch.load(f"models/odql/ODQL-{train_patterns}-{train_id}.pt")
                for train_id in range(n_train_evals)
            ]
            hist_state_dicts[train_patterns] = [
                torch.load(f"models/odql/HIST-{train_patterns}-{train_id}.pt")
                for train_id in range(n_train_evals)
            ]
    except:
        raise ("Failed to load models.")

    results = {
        "return": np.zeros((n_evals, n_train_evals, 8, 70)),
        "greedy_return": np.zeros((n_evals, n_train_evals, 8, 70)),
        "n_greedy_steps": np.zeros((n_evals, n_train_evals, 8, 70)),
    }
    for tps_id, train_patterns in enumerate(combinations(patterns, 4)):
        print(tps_id)
        for tid in range(n_train_evals):
            print(f"-> {tid}")
            for ep_id, eval_pattern in enumerate(patterns):
                for eid in range(n_evals):
                    env = IteratedLeverEnvironment(
                        payoffs=opt.payoffs,
                        n_iterations=opt.n_iterations + 1,
                        partner=FixedPatternPartner(eval_pattern),
                        include_step=False,
                        include_payoffs=False,
                    )

                    hist_rep = nn.LSTM(
                        input_size=len(env.dummy_obs()),
                        hidden_size=opt.hist_rep_dim,
                    )
                    hist_rep.load_state_dict(
                        hist_state_dicts[train_patterns][tid]
                    )

                    learner = DQNAgent(
                        q_net=nn.Sequential(
                            nn.Linear(opt.hist_rep_dim, opt.dqn_hidden_dim),
                            nn.ReLU(),
                            nn.Linear(opt.dqn_hidden_dim, env.n_actions()),
                        ),
                        capacity=opt.capacity,
                        batch_size=opt.batch_size,
                        lr=opt.dqn_lr,
                        tau=opt.tau,
                        len_update_cycle=opt.len_update_cycle,
                    )
                    learner.q_net.load_state_dict(
                        odql_state_dicts[train_patterns][tid]
                    )
                    learner.reset()

                    # WARMUP
                    eval_DQNAgent_xplay(
                        env=IteratedLeverEnvironment(
                            payoffs=opt.payoffs,
                            n_iterations=opt.n_iterations + 1,
                            include_step=False,
                            include_payoffs=False,
                        ),
                        agent1=learner,
                        agent2=deepcopy(learner),
                        hist_rep1=hist_rep,
                        hist_rep2=deepcopy(hist_rep),
                        bootstrap_last_step=True,
                        train1=True,
                        train2=True,
                        epsilon_schedule1=lambda step, episode_length: 0.3,
                        epsilon_schedule2=lambda step, episode_length: 0.3,
                    )

                    res_temp = eval_DQNAgent(
                        learner=learner,
                        hist_rep=hist_rep,
                        envs=[env],
                        bootstrap_last_step=True,
                        train=True,
                        epsilon=opt.epsilon,
                    )
                    results["return"][eid, tid, ep_id, tps_id] = res_temp[
                        "return"
                    ]
                    results["greedy_return"][
                        eid, tid, ep_id, tps_id
                    ] = res_temp["greedy_return"]
                    results["n_greedy_steps"][
                        eid, tid, ep_id, tps_id
                    ] = res_temp["n_greedy_steps"]

    torch.save(results, "results/odql-with-warumup-results.pickle")
