import torch
import torch.nn as nn
import torch.nn.functional as F

from levers import IteratedLeverEnvironment
from levers.partners import FixedPatternPartner
from levers.learner import DRQNAgent, DRQNetwork


TRUNCATED_LEN = 25
NUM_EPISODES=1000

# Initialize environment
env = IteratedLeverEnvironment(
    payoffs=[1., 1.], 
    n_iterations=TRUNCATED_LEN+1,
    partner=FixedPatternPartner([0, 1, 1, 1]),
    include_payoffs=False,
    include_step=False,
)

# Initialize DRQN agent
learner = DRQNAgent(
    q_net=DRQNetwork(
        input_size=len(env.dummy_obs()),
        hidden_size=4,
        n_actions=env.n_actions()
    ),
    capacity=8,
    batch_size=4,
    lr=0.01,
    gamma=1.0,
    len_update_cycle=4,
    tau=5e-4
)

for episode in range(NUM_EPISODES):
    # Reset environment and learner's hidden state and trajectory buffy
    obs = env.reset()
    learner.reset_trajectory_buffer(init_obs=obs)
    episode_return = 0

    # Step through environment
    # NOTE: To simulate open-ended environment, 
    #       we reset the environment after TRUNCATED_LEN steps
    for step in range(TRUNCATED_LEN):
        # Obtain action from learner and advance internal hidden state
        eps=0.3*max(0, 1 - 2*(episode + 1) / NUM_EPISODES)
        action = learner.act(obs, epsilon=eps)
        # Take step in environment
        next_obs, reward, done = env.step(action)
        # Add experience to learners trajectory buffer
        learner.update_trajectory_buffer(action, reward, next_obs, done)
        episode_return += reward
        # Update next observation -> observation
        obs = next_obs

    # Flush experience to replay memory and train
    learner.flush_trajectory_buffer()
    loss = learner.train()

    # Print current episode's stats
    print("Episode: {epi:4d} | Epsilon: {eps:2.2f} | Loss: {loss:6.4f} | Return: {ret:2.2f} {is_optimal}".format(
        epi=episode+1,
        eps=eps,
        loss=loss if loss else -1,
        ret=episode_return,
        is_optimal='*' if episode_return == TRUNCATED_LEN else '',
    ))