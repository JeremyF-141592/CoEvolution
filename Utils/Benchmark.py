import numpy as np
from Parameters import Configuration


def gramacy_lee(x):
    res = (x-1)**4 + np.sin(10*np.pi*x)/(2*x)
    maximum = 0.869011134989500
    return -res, maximum


class Benchmark:
    def __init__(self, size):
        self.benchmark_frequency = np.ones(size)
        self.benchmark_offset = np.zeros(size)
        self.size = size

    def __call__(self, agent, render=False, max_steps=2000, exceed_reward=0):
            wei = agent.get_weights()
            res = 0
            for i in range(len(wei)):
                value, maximum = Configuration.benchmark(self.benchmark_frequency[i] * wei[i] + self.benchmark_offset[i])
                res += value / maximum
            return 100.0 * res / len(wei)

    def get_child(self):
        child = Benchmark(self.size)
        child.benchmark_frequency = self.benchmark_frequency + np.random.uniform(-1, 1, size=self.size)
        child.benchmark_offset = self.benchmark_offset + np.random.uniform(-1, 1, size=self.size)
        return child

    def __getstate__(self):
        dic = dict()
        dic["Freq"] = self.benchmark_frequency.tolist()
        dic["Offset"] = self.benchmark_offset.tolist()
        dic["size"] = self.size
        return dic

    def __setstate__(self, state):
        self.__init__(state["size"])
        self.benchmark_offset = np.array(state["Offset"])
        self.benchmark_frequency = np.array(state["Freq"])
