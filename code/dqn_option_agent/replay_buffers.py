from collections import namedtuple
import random
import numpy as np

class BatchReplayMemory(object):

    def __init__(self, capacity):
        self.Transition = namedtuple('Transition',
                                     ('state', 'action', 'next_state', 'reward', 'terminate_flag','time_steps', 'valid_action_num'))
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class OptionReplayMemory:
    def __init__(self, capacity):
        self.Transition = namedtuple('Transition',
                                     ('state', 'action', 'next_state', 'reward', 'terminate_flag','time_steps', 'valid_action_num'))
        # list of tuple ==> f_function ==> t
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class TrajReplayMemory:
    def __init__(self, capacity):
        self.Transition = namedtuple('Transition',
                                     ('current_hex','next_hex'))
        # list of tuple ==> f_function ==> t
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
