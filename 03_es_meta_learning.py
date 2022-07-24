from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from levers import IteratedLeverEnvironment
from levers.partners import FixedPatternPartner
from levers.learner import OpenES, DQNAgent, Transition


def eval_learner(
    learner: DQNAgent,
    env: IteratedLeverEnvironment,
    n_episodes: int = 1
):
    """
    Evaluates the q-learning DQN-Agent `learner` in the environment `env`
    by rolling out `n_episodes` episodes. Cumulative reward serves as measure
    of fitness.
    """
    # Evaluate learners fitness
    cumulative_reward = 0
    for episode in range(n_episodes):
        # Reset environment
        obs = env.reset()
        done = False

        # Step through environment
        while not done:
            # Obtain action from learner
            action = learner.act(obs, 0.3)
            # Take step in environment
            next_obs, reward, done = env.step(action)
            cumulative_reward += reward
            # Give experience to learner and train
            learner.update_memory(Transition(obs, action, next_obs, reward, done))
            learner.train(done)
            # Update next observation -> observation
            obs = next_obs

    return cumulative_reward


def eval_population(
    population: List[DQNAgent],
    env: IteratedLeverEnvironment,
    n_episodes: int = 1
):
    """
    Evaluates the list of q-learning DQN-Agents `population` in the environment
    `env` by rolling out `n_episodes` episodes. Cumulative reward serves as
    measure of fitness.
    """
    population_fitness = []
    for params in population:
        # Populate learner with proposal params
        learner.reset(params)
        # Evaluate learner and save fitness
        fitness = eval_learner(learner, env, n_episodes)
        population_fitness.append(fitness)

    return population_fitness


class QNetwork(nn.Module):
    """Simple single hidden layer MLP with 4 hidden units. """

    def __init__(self, input_dim, n_actions):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4)
        self.fc2 = nn.Linear(4, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# Initialize environment
env = IteratedLeverEnvironment(
    payoffs=[1., 1., 1.], 
    n_iterations=6, 
    partner=FixedPatternPartner([0, 1, 2])
)

# Initialize DQN agent
learner = DQNAgent(
    q_net=QNetwork(input_dim=len(env.dummy_obs()), n_actions=env.n_actions()),
    capacity=16,
    batch_size=8,
    lr=0.01
)

# Initialize evolution strategy
es_strategy = OpenES(
    pop_size=20, 
    sigma_init=0.04, sigma_decay=0.999, sigma_limit=0.01, 
    optim_lr=0.01,
    optim_maximize=True,
)

# Futher settings
n_es_epochs = 20
n_q_learning_episodes = 100
print_every_k = 1

# Reset strategy and perform evolve using Ask-Eval-Tell loop
es_strategy.reset(learner.q_net.parameters())
for es_epoch in range(n_es_epochs):
    # Ask for proposal population
    population = es_strategy.ask()
    # Evaluate population
    population_fitness = eval_population(population, env, n_q_learning_episodes)
    # Tell (update)
    mean = es_strategy.tell(population_fitness)

    if (es_epoch + 1) % print_every_k == 0:
        print('ES-EPOCH: {epoch:2d} | REWARD (MIN/MEAN/MAX): {min:.2f}, {mean:.2f}, {max:.2f}'.format(
            epoch=es_epoch+1, 
            min=min(population_fitness),
            mean=sum(population_fitness) / es_strategy.pop_size,
            max=max(population_fitness)
        ))
        observations = [
            torch.tensor([0., 1., 1., 1., 0., 0., 0.]),
            torch.tensor([1., 1., 1., 1., 1., 0., 0.]),
            torch.tensor([2., 1., 1., 1., 0., 1., 0.]),
            torch.tensor([3., 1., 1., 1., 0., 0., 1.]),
            torch.tensor([4., 1., 1., 1., 1., 0., 0.]),
            torch.tensor([5., 1., 1., 1., 0., 1., 0.]),
        ]
        for obs in observations:
            q_vals = learner.q_net(obs)
            print('obs: {obs}, q-values: {q_vals}, greedy-action: {action}'.format(
                obs=obs, q_vals=q_vals, action=torch.argmax(q_vals)
            ))