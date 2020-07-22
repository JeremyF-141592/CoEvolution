from abc import ABC, abstractmethod


class Optimizer(ABC):
    @abstractmethod
    def step(self, gradient, state, args):
        """Returns a tuple (step taken in gradient space, new optimizer state)"""
        return NotImplementedError()

    @abstractmethod
    def default_state(self):
        """State is a dict, must include the key 't' representative of time elapsed"""
        return NotImplementedError()
