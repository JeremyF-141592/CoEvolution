import gym
from Parameters import Configuration
from abc import abstractmethod


class EnvironmentInterface(gym.Env):
    """"
    Allows a gym environment to be called like a function to run a whole episode.
    """

    def __call__(self, agent, render=False, max_steps=2000, exceed_reward=0):
        """
        An observer is a function acting on the path taken by the agent, returning an observation.
        A metric is a function returning the final score for a given agent, total reward and observation.
        """

        if Configuration.use_benchmark:
            wei = agent.get_weights()
            res = 0
            for w in wei:
                value, maximum = Configuration.benchmark(w)
                res += value / maximum
            return 100.0 * res / len(wei)

        total = 0
        for i in range(Configuration.nb_rounds):
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
                # path.append(state)
                fitness += reward
                count += 1
                if count > max_steps:
                    fitness += exceed_reward
                    break

            total += Configuration.metric(agent, self, fitness, Configuration.observer(path), Configuration.archive)
        return total

    @abstractmethod
    def get_child(self):
        return NotImplementedError

    @abstractmethod
    def __getstate__(self):
        return NotImplementedError

    @abstractmethod
    def __setstate__(self, state):
        return NotImplementedError
