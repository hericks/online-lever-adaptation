from itertools import combinations
from copy import deepcopy
from typing import Callable, Dict

import numpy as np
import random
import os

import torch
import torch.nn as nn

from levers import IteratedLeverEnvironment
from levers.evaluators.dqn_evaluators import eval_DQNAgent_xplay
from levers.helpers import generate_binary_patterns, visualize_xplay_matrix
from levers.learner import DQNAgent

from odql import get_parser


def warmup(
    env: IteratedLeverEnvironment,
    agent: DQNAgent,
    hist_rep: nn.RNNBase,
    epsilon_schedule: Callable,
):
    eval_DQNAgent_xplay(
        env=env,
        agent1=agent,
        agent2=deepcopy(agent),
        hist_rep1=hist_rep,
        hist_rep2=deepcopy(hist_rep),
        bootstrap_last_step=True,
        train1=True,
        train2=True,
        epsilon_schedule1=epsilon_schedule,
        epsilon_schedule2=epsilon_schedule,
    )


def print_episode(eval_stats: Dict):
    print(f"{xplay_patterns[id1]} playing with {xplay_patterns[id2]}.")
    n_steps = eval_stats["rewards"].shape[0]
    for step in range(n_steps):
        end = "" if step != n_steps - 1 else "\n"
        if eval_stats["greedy1"][step]:
            print("*", end=end)
        else:
            print(" ", end=end)
    for step in range(n_steps):
        end = "" if step != n_steps - 1 else "\n"
        print(int(eval_stats["action1"][step]), end=end)
    for step in range(n_steps):
        end = "" if step != n_steps - 1 else "\n"
        print(int(eval_stats["action2"][step]), end=end)
    for step in range(n_steps):
        end = "" if step != n_steps - 1 else "\n\n"
        if eval_stats["greedy2"][step]:
            print("*", end=end)
        else:
            print(" ", end=end)


if __name__ == "__main__":
    # Settings
    seed = 0
    n_agents = 70

    # Exploration strategies
    schedule1 = lambda step, episode_length: max(
        0.0, 1 * (1 - 4 * step / episode_length)
    )
    schedule2 = lambda step, episode_length: max(
        0.0, 1 * (1 - 4 * step / episode_length)
    )

    # Use models with warmup in ES
    es_warmup = False

    # Warmup before evaluation
    selfplay_warmup = True

    # Output path
    out_path = "results/crossplay/odql/"

    complete_component = "-complete" if n_agents == 70 else ""
    es_warmup_component = "-es-warmup" if es_warmup else ""
    sp_warmup_component = "-sp-warmup" if selfplay_warmup else ""
    name_template = (
        "odql-crossplay"
        + complete_component
        + es_warmup_component
        + sp_warmup_component
        + "-{n_agents}x{n_agents}-{seed}.png"
    )
    greedy_name_template = (
        "odql-crossplay"
        + complete_component
        + es_warmup_component
        + sp_warmup_component
        + "-{n_agents}x{n_agents}-{seed}-greedy.png"
    )

    # Models
    models_dir = "models/odql-es-warmup" if es_warmup else "models/odql"

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
    if n_agents < 70:
        xplay_patterns = random.sample(
            list(combinations(patterns, 4)), n_agents
        )
    else:
        xplay_patterns = list(combinations(patterns, 4))
    try:
        xplay_odql_state_dicts = [
            torch.load(os.path.join(models_dir, f"ODQL-{pattern}-0.pt"))
            for pattern in xplay_patterns
        ]
        xplay_hist_state_dicts = [
            torch.load(f"models/odql/HIST-{pattern}-0.pt")
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

    hist_rep1 = nn.LSTM(
        input_size=len(env.dummy_obs()),
        hidden_size=opt.hist_rep_dim,
    )
    hist_rep2 = deepcopy(hist_rep1)

    agent1 = DQNAgent(
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
    agent2 = deepcopy(agent1)

    xplay_results = np.zeros((n_agents, n_agents))
    xplay_greedy_results = np.zeros_like(xplay_results)
    for id1 in range(n_agents):
        print(id1)
        for id2 in range(n_agents):
            agent1.q_net.load_state_dict(xplay_odql_state_dicts[id1])
            agent2.q_net.load_state_dict(xplay_odql_state_dicts[id2])
            hist_rep1.load_state_dict(xplay_hist_state_dicts[id1])
            hist_rep2.load_state_dict(xplay_hist_state_dicts[id2])

            if selfplay_warmup:
                warmup(env, agent1, hist_rep1, schedule1)
                warmup(env, agent2, hist_rep2, schedule2)

            eval_stats = eval_DQNAgent_xplay(
                env=env,
                agent1=agent1,
                agent2=agent2,
                hist_rep1=hist_rep1,
                hist_rep2=hist_rep2,
                bootstrap_last_step=True,
                train1=True,
                train2=True,
                epsilon_schedule1=schedule1,
                epsilon_schedule2=schedule2,
            )

            rewards = eval_stats["rewards"]
            greedy_mask = eval_stats["greedy1"] & eval_stats["greedy2"]
            greedy_rewards = rewards[greedy_mask]
            n_greedy_steps = greedy_mask.sum()

            xplay_results[id1, id2] = rewards.sum() / (env.episode_length - 1)
            xplay_greedy_results[id1, id2] = (
                greedy_rewards.sum() / n_greedy_steps
            )

    visualize_xplay_matrix(
        xplay_greedy_results,
        os.path.join(
            out_path,
            greedy_name_template.format(n_agents=n_agents, seed=seed),
        ),
        vmin=0,
        vmax=1,
    )
    visualize_xplay_matrix(
        xplay_results,
        os.path.join(
            out_path, name_template.format(n_agents=n_agents, seed=seed)
        ),
        vmin=0,
        vmax=1,
    )
