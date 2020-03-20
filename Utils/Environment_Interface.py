import gym
from Utils.Observers import empty_observer
from Utils.Metrics import fitness_metric


class EnvironmentInterface(gym.Env):
    """"
    Custom class representing an environment.
    Allows an environment to be called like a function to run a simulation.
    """
    def __call__(self, agent, observer=empty_observer, metric=fitness_metric, *args, **kwargs):
        """
        An observer is a function acting on the path taken by the agent, returning an observation.
        A metric is a function returning the final score for a given agent, total reward and observation.
        """
        state = self.reset()
        done = False

        total_reward = 0
        path = list()
        while not done:
            action = agent.choose_action(state)
            state, reward, done, info = self.step(action)
            path.append(state)
            total_reward += reward

        return metric(agent, total_reward, observer(path))
