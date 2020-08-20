import numpy as np
from Parameters import Configuration
from ABC.Agents import Agent, AgentFactory
from ABC.Environments import Environment, EnvironmentFactory


def cross_cosinus_gaussian(x, y):
    sigma = 0.2
    gauss = np.exp(-x**2 / (2*sigma**2))*2 - 1
    crossed = np.cos(np.pi * (-abs(x) + abs(y)))
    return 100*max(gauss, crossed / (abs(0.2*x) + 1))


def diag_gaussian(x, y):
    return 33.33*(np.exp(-9*x**2) + 2*np.exp(-1/9.0 * (x-y)**2))


def cross_gaussian(x, y):
    sigma = 0.2
    gauss = np.exp(-(x+y)**2 / (2*sigma**2))*2 - 1
    crossed = np.cos(np.pi * (x - y))
    return 100*max(gauss, crossed / (abs(0.2*(x+y)) + 1))


def pata_ec_test(x, y):
    sigma = 0.2
    gauss = np.exp(-x**2 / (2*sigma**2))*2 - 1
    crossed = np.cos(np.pi * (-abs(x) + abs(y)))
    if 12 > y > 11:
        return 200*np.random.random() - 100
    return 100*max(gauss, crossed / (abs(0.2*x) + 1))


def cross_cosinus(x, y):
    crossed = np.cos(np.pi * (-abs(x) + abs(y)))
    return 100*crossed / (abs(0.2*x) + 1)


def cosx(x, y):
    return 100*np.cos(np.pi * abs(x)) / (abs(0.2*x) + 1)


class BenchmarkEnv(Environment):
    def __init__(self, y):
        self.y_value = y

        # ------------------------------------------------------------------------------------------- Benchmark Function
        self.map = diag_gaussian
        # --------------------------------------------------------------------------------------------------------------

    def __call__(self, agent, render=False, max_steps=2000, exceed_reward=0):
        return Configuration.metric(agent, self, self.map(agent.value, self.y_value), [agent.value])

    def get_child(self):
        child = BenchmarkEnv(self.y_value + np.random.uniform(-3, 3))
        return child

    def crossover(self, other):
        child = BenchmarkEnv((self.y_value + other.y_value) / 2.0)
        return child

    def __getstate__(self):
        dic = dict()
        dic["y"] = self.y_value
        return dic

    def __setstate__(self, state):
        self.__init__(state["y"])


class BenchmarkEnvFactory(EnvironmentFactory):
    def __init__(self, y_init):
        self.y = y_init

    def new(self):
        return BenchmarkEnv(self.y)


class BenchmarkAg(Agent):
    def __init__(self):
        self.value = np.random.uniform(-50.0, 50.0)
        self.opt_state = None

    def choose_action(self, state):
        return self.value

    def randomize(self):
        self.value = np.random.uniform(-40.0, 40.0)

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
    k = np.linspace(-20, 20, num=size)
    a = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            a[j, i] = diag_gaussian(k[i], k[j])
    plt.imshow(a, cmap='hot', interpolation='nearest')
    plt.show()
