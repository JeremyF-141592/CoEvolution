import numpy as np
from Parameters import Configuration


def gramacy_lee(x):
    res = (x-1)**4 + np.sin(10*np.pi*x)/(2*x)
    return -res


class Benchmark:
    def __init__(self, config):
        size, maximum, argmax = config
        self.benchmark_frequency = np.ones(size)
        self.benchmark_offset = np.zeros(size)
        self.size = size
        self.max = maximum
        self.argmax = argmax

    def __call__(self, agent, render=False, max_steps=2000, exceed_reward=0):
            wei = agent.get_weights()
            res = 0
            for i in range(len(wei)):
                value = Configuration.benchmark(self.benchmark_frequency[i] * wei[i] + self.benchmark_offset[i])
                res += value / self.max
            return 100.0 * res / len(wei)

    def get_child(self):
        child = Benchmark((self.size, self.max, self.argmax))
        child.benchmark_frequency = self.benchmark_frequency + np.random.uniform(-0.01, 0.01, size=self.size)
        child.benchmark_offset = self.benchmark_offset + np.random.uniform(-0.01, 0.01, size=self.size)
        return child

    def mate(self, other):
        child = Benchmark((self.size, self.max, self.argmax))
        child.benchmark_frequency = (self.benchmark_frequency + other.benchmark_frequency)/2.0
        child.benchmark_offset = (self.benchmark_offset + other.benchmark_offset)/2.0
        return child

    def __getstate__(self):
        dic = dict()
        dic["Freq"] = self.benchmark_frequency.tolist()
        dic["Offset"] = self.benchmark_offset.tolist()
        dic["size"] = self.size
        dic["max"] = self.max
        dic["argmax"] = self.argmax
        return dic

    def __setstate__(self, state):
        self.__init__((state["size"], state["max"], state["argmax"]))
        self.benchmark_offset = np.array(state["Offset"])
        self.benchmark_frequency = np.array(state["Freq"])

