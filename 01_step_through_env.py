from levers import IteratedLeverEnvironment
from levers.partners import RandomPartner, FixedPatternPartner


# Environment parameters
payoffs = [1., 1.]
n_iterations = 10

# Use partner playing randomly...
partner = RandomPartner()
# ... or a partner playing a fixed pattern
partner = FixedPatternPartner([0, 1, 0, 0])

# Initialize environment
env = IteratedLeverEnvironment(payoffs, n_iterations, partner)

# Reset environment
obs = env.reset()
print(f'obs: {obs} (initial)')

# Step through environment
done = False
while not done:
    action = 0
    obs, reward, done = env.step(action)
    print(f'obs: {obs}, reward: {reward}, done: {done}') 