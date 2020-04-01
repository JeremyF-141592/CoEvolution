import gym
from Parameters import Configuration
from abc import abstractmethod


class EnvironmentInterface(gym.Env):
    """"
    Allows a gym environment to be called like a function to run a whole episode.
    """
    def __call__(self, agent, render=False, max_steps=1000, exceed_reward=-100):
        """
        An observer is a function acting on the path taken by the agent, returning an observation.
        A metric is a function returning the final score for a given agent, total reward and observation.
        """
        state = self.reset()
        done = False

        total_reward = 0
        path = list()
        count = 0
        while not done:
            if render:
                self.render()

            action = agent.choose_action(state)
            state, reward, done, info = self.step(action)
            # path.append(state)
            total_reward += reward

            count += 1
            if count > max_steps:
                total_reward += exceed_reward
                break

        return Configuration.metric(agent.__getstate__()["as_vector"], self.__getstate__()["as_vector"], 
									total_reward, Configuration.observer(path), Configuration.archive)

    @abstractmethod
    def __getstate__(self):
        return NotImplementedError

    @abstractmethod
    def __setstate__(self, state):
        return NotImplementedError
