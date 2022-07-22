import random

from collections import deque, namedtuple


Transition = namedtuple(
    'Transition',
    ('state', 'action', 'next_state', 'reward', 'done')
)


class ReplayMemory():
    """
    Replay memory to store state-action-state transitions used for q-learning.
    """

    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, transition: Transition):
        """Pushes transition to the replay memory. """
        self.memory.append(transition)

    def sample(self, batch_size: int):
        """Returns sample (of size batch_size) of replay memory. """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)