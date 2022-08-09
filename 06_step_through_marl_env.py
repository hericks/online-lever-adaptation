from levers import IteratedLeverEnvironment
from random import randint


# Environment parameters
payoffs = [1., 1.]
n_iterations = 10

# Initialize environment without lever game partner
env = IteratedLeverEnvironment(payoffs, n_iterations)

# Reset environment
joint_obs = env.reset()
obs1 = joint_obs[0,]
obs2 = joint_obs[1,]
print(f'(player 1) obs: {obs1}, (player 2) obs {obs2} (initial)')

# Step through environment
done = False
while not done:
    action1 = randint(0, 1)
    action2 = randint(0, 1)
    joint_obs, reward, done = env.step([action1, action2])
    obs1 = joint_obs[0,]
    obs2 = joint_obs[1,]
    print(f'(player 1) obs: {obs1}, (player 2) obs: {obs2}, reward: {reward}, done: {done}')