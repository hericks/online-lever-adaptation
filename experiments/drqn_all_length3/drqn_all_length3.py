# Relative imports outside of package
import sys
from os import path
sys.path.insert(1, path.join(sys.path[0], '../..'))

from levers import IteratedLeverEnvironment
from levers.partners import FixedPatternPartner
from levers.learner import DRQNAgent, DRQNetwork

import pickle
import random
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

# Training settings
num_episodes = 1000
epsilon = 0.3

# 
n_evaluations = 10

# Patterns
patterns = [
    (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
    (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1),
]


results = {}
for train_patterns in itertools.combinations(patterns, 4):
    print(f'TRAIN-PATTERN: {train_patterns}')

    # Construct test pattern from train pattern
    test_patterns = tuple(p for p in patterns if p not in train_patterns)

    # Construct list of environments to train on
    train_envs = [
        IteratedLeverEnvironment(
            payoffs, truncated_length+1, FixedPatternPartner(list(pattern)),
            include_step, include_payoffs)
        for pattern in train_patterns
    ]

    # Setup temporary results dict
    train_patterns_results = {}
    for pattern in patterns:
        train_patterns_results[pattern] = []

    # Fill temporary results dict
    for i in range(n_evaluations):
        print(f'-> evaluation: {i+1} / {n_evaluations}')

        # Reset learner
        learner = DRQNAgent(
            DRQNetwork(
                input_size=len(train_envs[0].dummy_obs()),
                hidden_size=hidden_size,
                n_actions=train_envs[0].n_actions()),
            capacity, batch_size, lr, gamma, len_update_cycle, tau
        )

        # Train learner
        for episode in range(num_episodes):
            # Sample reset environment from training environments
            env = random.sample(train_envs, 1)[0]
            obs = env.reset()
            learner.reset_trajectory_buffer(init_obs=obs)

            # Step through environment
            for step in range(truncated_length):
                action = learner.act(obs, epsilon)
                next_obs, reward, done = env.step(action)
                learner.update_trajectory_buffer(action, reward, next_obs, done)
                obs = next_obs 

            # Flush experience to replay memory and train learner
            learner.flush_trajectory_buffer()
            learner.train()

        # Evaluate learner on each possible partner pattern
        for pattern in patterns:
            eval_env = IteratedLeverEnvironment(
                payoffs, truncated_length+1, FixedPatternPartner(list(pattern)),
                include_step, include_payoffs)

            ret = 0
            obs = eval_env.reset()
            for step in range(truncated_length):
                action = learner.act(obs)
                next_obs, reward, done = eval_env.step(action)
                ret += reward
                obs = next_obs 

            train_patterns_results[pattern].append(ret)

    print("Results:")
    for pattern in patterns:
        print(f'-> pattern: {pattern}, returns: {train_patterns_results[pattern]}')
    print('')

    # Flush current result and save to disk
    results[train_patterns] = train_patterns_results
    data_folder = path.join(path.dirname(__file__), 'data')
    out_file = open(path.join(data_folder, 'results.pkl'), 'wb')
    pickle.dump(results, out_file)
    out_file.close()