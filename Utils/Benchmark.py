import numpy as np
from Parameters import Configuration
from Utils.Agents import Agent, AgentFactory


def env_map(x, y):
    return 100 * (np.cos(2. * np.pi * x) / (1. + 0.1*abs(x * y)))


def convoluted_map(x, y):
    slide = 0.2*np.cos(15*np.pi*x*(1/(abs(y)+1)))
    return np.cos(2 * np.pi * x)/(1.+abs(x) + 0.5 * abs(y)) + slide


class Benchmark:
    def __init__(self, config):
        self.y_value = config
        self.map = Configuration.benchmark

    def __call__(self, agent, render=False, max_steps=2000, exceed_reward=0):
        return Configuration.benchmark(agent.value, self.y_value)

    def get_child(self):
        child = Benchmark(self.y_value + np.random.uniform(-1, 1))
        return child

    def mate(self, other):
        child = Benchmark((self.y_value + other.y_value) / 2.0)
        return child

    def __getstate__(self):
        dic = dict()
        dic["y"] = self.y_value
        return dic

    def __setstate__(self, state):
        self.__init__(state["y"])


class BenchmarkAg(Agent):
    def __init__(self):
        self.value = np.random.uniform(-100.0, 100.0)
        self.opt_state = Configuration.optimizer.default_state()

    def choose_action(self, state):
        return self.value

    def randomize(self):
        return np.random.uniform(-100.0, 100.0)

    def get_weights(self):
        return np.array([self.value])

    def set_weights(self, weights):
        self.value = weights[0]

    def get_opt_state(self):
        return self.opt_state

    def set_opt_state(self, state):
        self.opt_state = state

    def __getstate__(self):
        dic = dict()
        dic["value"] = self.value
        dic["opt"] = self.opt_state
        return dic

    def __setstate__(self, state):
        self.__init__()
        self.value = state["value"]
        self.opt_state = state["opt"]


class BenchmarkFactory(AgentFactory):
    def new(self):
        return BenchmarkAg()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    size = 200
    k = np.linspace(-2, 2, num=size)
    a = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            a[j, i] = convoluted_map(k[i], k[j])
    plt.imshow(a, cmap='hot', interpolation='nearest')
    plt.show()