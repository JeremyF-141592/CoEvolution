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


class AgentFactory(ABC):
    @abstractmethod
    def new(self):
        """Must return an Agent object"""
        return NotImplementedError



