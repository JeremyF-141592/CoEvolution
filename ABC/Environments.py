import gym
from Parameters import Configuration
from abc import abstractmethod, ABC


class Environment(ABC):
    @abstractmethod
    def __call__(self, agent, *args, **kwargs):
        """Runs an entire episode of one agent over this environment."""
        return NotImplementedError

    @abstractmethod
    def get_child(self):
        return NotImplementedError

    @abstractmethod
    def crossover(self, other):
        return NotImplementedError

    @abstractmethod
    def __getstate__(self):
        return NotImplementedError

    @abstractmethod
    def __setstate__(self, state):
        return NotImplementedError


class GymInterface(gym.Env, ABC):
    def __call__(self, agent, render=False, use_state_path=False, max_steps=2000, exceed_reward=0):
        """
        Runs an entire episode of one agent over this environment.

        This method works in compliance with gym.Env methods :
            .reset()
            .render()
            .step(action)

        The execution stops at 'max_steps' iterations if not stopped otherwise, giving 'exceed_reward' to the agent.
        An observer is a function acting on the path taken by the agent, returning an observation.
        """
        state = self.reset()
        done = False

        fitness = 0
        path = list()
        count = 0
        while not done:
            if render:
                self.render()

            action = agent.choose_action(state)
            state, reward, done, info = self.step(action)

            if use_state_path:
                path.append(state)

            fitness += reward
            count += 1
            if count > max_steps:
                fitness += exceed_reward
                break

        return Configuration.metric(agent, self, fitness, path)


class EnvironmentFactory(ABC):
    @abstractmethod
    def new(self):
        """Returns an Environment object."""
        return NotImplementedError
