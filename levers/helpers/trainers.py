from typing import List

import random

from .. import IteratedLeverEnvironment
from ..learner import DRQNAgent


def train_drqn_agent(
    agent: DRQNAgent,
    envs: List[IteratedLeverEnvironment],
    n_episodes: int,
    epsilon: float,
    bootstrap_last_step: bool,
):
    """
    Trains a DRQN agent in environments uniformly sampled from `envs` for
    `n_episodes` episodes with exploration probability `epsilon`.
    """
    train_stats = {
        "episode": [],
        "loss": [],
        "return": [],
    }
    for episode in range(n_episodes):
        # Sample and reset environment from training environments
        env = random.sample(envs, 1)[0]
        obs = env.reset()
        agent.reset_new_episode(init_obs=obs)

        # Setup stats
        episode_return = 0

        # Step through environment
        for step in range(env.episode_length - int(bootstrap_last_step)):
            action = agent.act(obs, epsilon)
            next_obs, reward, done = env.step(action)
            agent.update_trajectory_buffer(action, reward, next_obs, done)
            obs = next_obs

            # Update episode stats
            episode_return += reward

        # Flush experience to replay memory and train learner
        agent.flush_trajectory_buffer()
        loss = agent.train()

        # Fill train stats
        train_stats["episode"].append(episode)
        train_stats["loss"].append(loss)
        train_stats["return"].append(episode_return)

    return train_stats
