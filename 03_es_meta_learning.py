from typing import List, Dict

import random
import numpy as np

import torch
import torch.nn as nn

from torch.nn.utils import vector_to_parameters, parameters_to_vector

from evotorch import Problem
from evotorch.logging import StdOutLogger

from levers import IteratedLeverEnvironment
from levers.partners import FixedPatternPartner
from levers.learner import DQNAgent, Transition
from levers.learner.open_es import OpenES


def eval_learner(
    param_vec: torch.Tensor,
    learner: DQNAgent,
    env: IteratedLeverEnvironment,
    n_episodes: int,
):
    """
    Evaluates the q-learning DQN-Agent `learner` in the environment `env`
    by rolling out `n_episodes` episodes. Cumulative reward serves as measure
    of fitness.
    """
    # Load learner's state
    vector_to_parameters(param_vec, learner.q_net.parameters())
    learner.target_net.load_state_dict(learner.q_net.state_dict())
    learner.reset()

    # Evaluate learners fitness
    cumulative_reward = 0
    for episode in range(n_episodes):
        # Reset environment
        obs = env.reset()
        done = False

        # Step through environment
        while not done:
            # Obtain action from learner
            action, _ = learner.act(obs, epsilon=0.3)
            # Take step in environment
            next_obs, reward, done = env.step(action)
            cumulative_reward += reward
            # Give experience to learner and train
            transition = Transition(obs, action, next_obs, reward, done)
            learner.update_memory(transition)
            learner.train()
            # Update next observation -> observation
            obs = next_obs

    return cumulative_reward


# Reproducibility
seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Initialize environment
env = IteratedLeverEnvironment(
    payoffs=[1., 1., 1.], 
    n_iterations=6, 
    partner=FixedPatternPartner([0, 1, 2]),
    include_payoffs=False,
)

# Initialize DQN agent
learner = DQNAgent(
    q_net=nn.Sequential(
        nn.Linear(len(env.dummy_obs()), 4),
        nn.ReLU(),
        nn.Linear(4, env.n_actions())
    ),
    capacity=16,
    batch_size=8,
    lr=0.01,
    tau=1.0,
    len_update_cycle=10*env.episode_length,
)

# Initialize ES problem and algorithm
problem = Problem(
    'max',
    lambda param_vec: eval_learner(param_vec, learner, env, 10),
    solution_length=sum(p.numel() for p in learner.q_net.parameters()),
    initial_bounds=(-1, 1),
    num_actors='max'
)
searcher = OpenES(
    problem,
    popsize=50,
    learning_rate=0.1,
    stdev_init=0.5,
    stdev_decay=0.99,
    stdev_min=0.1,
    mean_init=parameters_to_vector(learner.q_net.parameters())
)

# Attach logger to ES algorithm and run
logger = StdOutLogger(searcher)
searcher.run(100)