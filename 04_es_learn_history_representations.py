from typing import List

import torch
import torch.nn as nn

from torch.nn.utils import vector_to_parameters, parameters_to_vector

from evotorch import Problem
from evotorch.logging import StdOutLogger

from levers import IteratedLeverEnvironment
from levers.helpers.helpers import n_total_parameters
from levers.learner.history_shaper import HistoryShaper
from levers.partners import FixedPatternPartner
from levers.learner import OpenES, DQNAgent, Transition


def eval_learner(
    param_vec: torch.Tensor,
    learner: DQNAgent,
    hist_rep: HistoryShaper,
    env: IteratedLeverEnvironment,
    n_episodes: int = 1
):
    """
    Evaluates the q-learning DQN-Agent `learner` in the environment `env`
    by rolling out `n_episodes` episodes. Cumulative reward serves as measure
    of fitness.
    """
    n_learner_params = sum(p.numel() for p in learner.q_net.parameters())

    # Load learner's state
    vector_to_parameters(
        param_vec[:n_learner_params],
        learner.q_net.parameters()
    )
    learner.reset()

    # Load history representation's state
    vector_to_parameters(
        param_vec[n_learner_params:],
        hist_rep.net.parameters()
    )

    # Evaluate learners fitness
    cumulative_reward = 0
    for episode in range(n_episodes):
        # Reset environment
        obs = env.reset()
        obs_rep, hidden = hist_rep.net(obs.unsqueeze(0))
        done = False

        # Step through environment
        while not done:
            # Obtain action from learner
            action, _ = learner.act(obs_rep.squeeze(0), epsilon=0.3)

            # Take step in environment
            next_obs, reward, done = env.step(action)
            cumulative_reward += reward

            # Compute history representation
            next_obs_rep, next_hidden = hist_rep.net(
                next_obs.unsqueeze(0), hidden)

            # Give experience to learner and train
            learner.update_memory(
                Transition(
                    obs_rep.squeeze(0).detach(),
                    action, 
                    next_obs_rep.squeeze(0).detach(), 
                    reward, done
                )
            )
            learner.train()

            # Update next observation -> observation
            obs_rep = next_obs_rep
            hidden = next_hidden

    return cumulative_reward


# Initialize environment
env = IteratedLeverEnvironment(
    payoffs=[1., 1., 1.], 
    n_iterations=6, 
    partner=FixedPatternPartner([0, 1, 2]),
    include_step=False,
    include_payoffs=False,
)

# Initialize history shaper with LSTM net
hist_rep_output_size=4
hist_rep = HistoryShaper(
    hs_net=nn.LSTM(input_size=len(env.dummy_obs()),
    hidden_size=hist_rep_output_size),
)

# Initialize DQN agent
learner = DQNAgent(
    q_net=nn.Sequential(
        nn.Linear(hist_rep_output_size, 4),
        nn.ReLU(),
        nn.Linear(4, env.n_actions())
    ),
    capacity=16,
    batch_size=8,
    lr=0.01
)

# Initialize ES problem and algorithm
n_learner_params = n_total_parameters(learner.q_net)
n_hist_rep_params = n_total_parameters(hist_rep.net)

problem = Problem(
    'max',
    lambda param_vec: eval_learner(param_vec, learner, hist_rep, env, 20),
    solution_length=n_learner_params + n_hist_rep_params,
    initial_bounds=(-1, 1),
    num_actors='max',
)

searcher = OpenES(
    problem,
    popsize=50,
    learning_rate=0.05,
    stdev_init=0.1,
    stdev_decay=0.999,
    stdev_min=0.01,
    mean_init=torch.concat((
        parameters_to_vector(learner.q_net.parameters()),
        parameters_to_vector(hist_rep.net.parameters())
    ))
)

# Attach logger to ES algorithm and run
logger = StdOutLogger(searcher)
searcher.run(150)