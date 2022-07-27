import torch
import torch.nn as nn
import torch.nn.functional as F

from levers import IteratedLeverEnvironment
from levers.partners import FixedPatternPartner
from levers.learner import DQNAgent, Transition


# Initialize environment
env = IteratedLeverEnvironment(
    payoffs=[1., 1.], 
    n_iterations=2, 
    partner=FixedPatternPartner([0])
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
    lr=0.005
)

for episode in range(200):
    # Reset environment
    obs = env.reset()
    done = False

    # Step through environment
    while not done:
        # Obtain action from learner
        action = learner.act(obs, epsilon=0.5)
        # Take stap in environment
        next_obs, reward, done = env.step(action)
        # Give experience to learner and train
        learner.update_memory(Transition(obs, action, next_obs, reward, done))
        learner.train(done)
        # Update next observation -> observation
        obs = next_obs

    # Evaluate learner every 50 episodes
    if (episode+1) % 50 == 0:
        print(f'--- AFTER EPISODE {episode+1}')
        observations = [
            torch.tensor([0., 1., 1., 0., 0.]),
            torch.tensor([1., 1., 1., 1., 0.]) 
        ]
        for obs in observations:
            print(f'obs: {obs}, q-values: {learner.q_net(obs)}')