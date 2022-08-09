from ntpath import join
from os import path
from copy import deepcopy

from levers import IteratedLeverEnvironment
from levers.learner import HistoryShaper, DQNAgent, Transition

import torch
import torch.nn as nn

# Environment parameters
payoffs = [1., 1.]
n_iterations = 100

# Initialize environment without lever game partner
env = IteratedLeverEnvironment(
    payoffs,
    n_iterations+1, 
    include_payoffs=False,
    include_step=False
)

# Initialize history shaper with LSTM net
hs_output_size=4
hs1 = HistoryShaper(
    hs_net=nn.LSTM(input_size=len(env.dummy_obs()[0,]), hidden_size=hs_output_size)
)
hs2 = deepcopy(hs1)

# Initialize DQN agent
l1 = DQNAgent(
    q_net=nn.Sequential(
        nn.Linear(hs_output_size, 4),
        nn.ReLU(),
        nn.Linear(4, env.n_actions())
    ),
    capacity=16,
    batch_size=8,
    lr=0.01
)
l2 = deepcopy(l1)

# Load the models
models_dir = 'experiments/online_qlearner_length3/data'

model_name1 = '-net-pattern=((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1))-eval_id=00.pt'
hs1.net.load_state_dict(torch.load(path.join(models_dir, 'hs' + model_name1)))
l1.q_net.load_state_dict(torch.load(path.join(models_dir, 'q' + model_name1)))
l1.reset()

model_name2 = '-net-pattern=((0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1))-eval_id=00.pt'
hs2.net.load_state_dict(torch.load(path.join(models_dir, 'hs' + model_name1)))
l2.q_net.load_state_dict(torch.load(path.join(models_dir, 'q' + model_name1)))
l2.reset()

# Reset environment
joint_obs = env.reset()

obs1 = joint_obs[0,]
obs_rep1, hidden1 = hs1.net(obs1.unsqueeze(0))

obs2 = joint_obs[1,]
obs_rep2, hidden2 = hs2.net(obs1.unsqueeze(0))

# Step through environment
for step in range(n_iterations):
    # Obtain action from learners
    epsilon = 0.2 * max(0, 1 - 2 * step / n_iterations)
    action1, _ = l1.act(obs_rep1.squeeze(0), epsilon=epsilon)
    action2, _ = l1.act(obs_rep2.squeeze(0), epsilon=epsilon)

    # Take step in environment
    joint_next_obs, reward, done = env.step([action1, action2])
    next_obs1 = joint_next_obs[0]
    next_obs2 = joint_next_obs[1]

    # Compute history representation
    next_obs_rep1, next_hidden1 = hs1.net(next_obs1.unsqueeze(0), hidden1)
    next_obs_rep2, next_hidden2 = hs2.net(next_obs2.unsqueeze(0), hidden2)

    # Give experience to learner and train
    l1.update_memory(
        Transition(
            obs_rep1.squeeze(0).detach(),
            action1, 
            next_obs_rep1.squeeze(0).detach(), 
            reward, done
        )
    )
    l1.train(done)
    l2.update_memory(
        Transition(
            obs_rep2.squeeze(0).detach(),
            action2, 
            next_obs_rep2.squeeze(0).detach(), 
            reward, done
        )
    )
    l2.train(done)

    # Update next observation -> observation
    obs_rep1 = next_obs_rep1
    obs_rep2 = next_obs_rep2
    hidden1 = next_hidden1
    hidden2 = next_hidden2

    print(f'STEP {step:2d} | ACTION (1/2): {action1}/{action2} | REWARD: {reward}')