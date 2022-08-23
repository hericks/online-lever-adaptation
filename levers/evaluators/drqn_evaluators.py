from levers import IteratedLeverEnvironment
from levers.learner import DRQNAgent

import torch
import torch.nn as nn


def eval_DRQNAgent_xplay(
    env: IteratedLeverEnvironment,
    agent1: DRQNAgent,
    agent2: DRQNAgent,
    bootstrap_last_step: bool = True,
) -> float:
    joint_obs = env.reset()
    agent1.reset()
    agent2.reset()

    episode_return = 0
    for step in range(env.episode_length - int(bootstrap_last_step)):
        action1 = agent1.act(joint_obs[0])
        action2 = agent2.act(joint_obs[1])
        next_joint_obs, reward, _ = env.step([action1, action2])
        episode_return += reward
        joint_obs = next_joint_obs

    return episode_return
