# /usr/bin/env python2
import random
from collections import namedtuple
Transition = namedtuple('Transition',
                   ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    """
    This class is to store the observations for training of Q-function.
    For detailed explanation (code is taken from here), refer: http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#replay-memory

    Parameters:
    ______________
        capacity: int
            Defines the maximum number of observations to hold in the list
    """

    def __init__(self, capacity):
        """ Instantiate this class with memory of the size of capacity and a pointer to the list"""
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """
        It stores the observations in the list: memory.

        Parameters:
        ______________
            *args: tuple of 4 elements
                It assumes the form: (state, action, next_state, reward)

        Returns:
        ______________
            None
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position+1)%self.capacity

    def sample(self, batch_size):
        """
        Gets the random sample from the observation

        Parameters:
        ______________
            batch_size: Number of observtations to return

        Returns:
        ______________
            sample: list[namedtuples]
                each element is a namedtuple of Transition

        """
        if batch_size >= len(self.memory):
            return self.memory
        return random.sample(self.memory, batch_size)

    def get_average_reward(self):
        rewards = [x.reward for x in self.memory]
        return 1.0*sum(rewards)/len(rewards)

    def __len__(self):
        return len(self.memory)