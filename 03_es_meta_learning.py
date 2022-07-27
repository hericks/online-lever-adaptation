from typing import List, Dict

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
            action = learner.act(obs, epsilon=0.3)
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
    population: Dict[str, List[DQNAgent]],
    env: IteratedLeverEnvironment,
    n_episodes: int = 1
):
    """
    Evaluates the list of parameters for a q-learning DQN-Agent `population`
    in the environment `env` by rolling out `n_episodes` episodes.
    Cumulative reward serves as measure of fitness.
    """
    population_fitness = []
    for member in population:
        # Populate learner with proposal params
        learner.reset(member['q_net'])
        # Evaluate learner and save fitness
        fitness = eval_learner(learner, env, n_episodes)
        population_fitness.append(fitness)

    return population_fitness


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
    lr=0.01
)

# Initialize evolution strategy
es_strategy = OpenES(
    pop_size=50, 
    sigma_init=0.1, sigma_decay=0.999, sigma_limit=0.01, 
    optim_lr=0.05,
    optim_maximize=True,
)

# Further settings
n_es_epochs = 50
n_q_learning_episodes = 10
print_every_k = 1

# Reset strategy and perform evolve using Ask-Eval-Tell loop
es_strategy.reset({'q_net': learner.q_net.parameters()})
for es_epoch in range(n_es_epochs):
    # Ask for proposal population
    population = es_strategy.ask()
    # Evaluate population
    population_fitness = eval_population(population, env, n_q_learning_episodes)
    # Tell (update)
    mean = es_strategy.tell(population_fitness)

    if (es_epoch + 1) % print_every_k == 0:
        learner.reset(mean['q_net'])
        observations = [
            torch.tensor([0., 0., 0., 0.]),
            torch.tensor([1., 1., 0., 0.]),
            torch.tensor([2., 0., 1., 0.]),
            torch.tensor([3., 0., 0., 1.]),
            torch.tensor([4., 1., 0., 0.]),
            torch.tensor([5., 0., 1., 0.]),
        ]
        greedy_pattern = [
            torch.argmax(learner.q_net(obs)).item() for obs in observations]
        print('ES-EPOCH: {epoch:2d} | REWARD (MIN/MEAN/MAX): {min:.2f}, {mean:.2f}, {max:.2f} | GREEDY-PATTERN: {pattern}'.format(
            epoch=es_epoch+1, 
            min=min(population_fitness),
            mean=sum(population_fitness) / es_strategy.pop_size,
            max=max(population_fitness),
            pattern=greedy_pattern
        ))