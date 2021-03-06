from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def choose_action(self, state):
        return NotImplementedError

    @abstractmethod
    def randomize(self):
        return NotImplementedError

    @abstractmethod
    def get_weights(self):
        return NotImplementedError

    @abstractmethod
    def set_weights(self, weights):
        return NotImplementedError

    @abstractmethod
    def get_opt_state(self):
        """Communicate with the optimizer."""
        return NotImplementedError

    @abstractmethod
    def set_opt_state(self, state):
        return NotImplementedError

    #  Needed for Pickle

    @abstractmethod
    def __getstate__(self):
        return NotImplementedError

    @abstractmethod
    def __setstate__(self, state):
        return NotImplementedError


class AgentFactory(ABC):
    @abstractmethod
    def new(self):
        """Returns an Agent object"""
        return NotImplementedError
