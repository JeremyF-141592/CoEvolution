import numpy as np
from Parameters import Configuration
from Templates.Agents import Agent, AgentFactory
from Templates.Environments import Environment, EnvironmentFactory


class KNNBenchmarkEnv(Environment):

    def __init__(self, dimension, spec_fit, gen_fit, gen_points, bounds):
        self.dim = dimension
        self.bounds = bounds
        self.generalist_points = gen_points
        self.generalist_fit = gen_fit

        self.specialist_points = np.random.uniform(bounds[0], bounds[1], size=(len(spec_fit), dimension))
        self.specialist_fit = spec_fit

    def __call__(self, agent, render=False, max_steps=2000, exceed_reward=0):
        fit = 0
        dist = 0
        for i in range(len(self.generalist_points)):
            d_inv = (np.linalg.norm(agent.value - self.generalist_points[i]) + 1e-6)**-2
            dist += d_inv
            fit += d_inv * self.generalist_fit[i]
        for i in range(len(self.specialist_points)):
            d_inv = (np.linalg.norm(agent.value - self.specialist_points[i]) + 1e-6)**-2
            dist += d_inv
            fit += d_inv * self.specialist_fit[i]

        fit /= dist
        return Configuration.metric(agent, self, fit, [])

    def get_child(self):
        child = KNNBenchmarkEnv(self.dim, self.specialist_fit, self.generalist_fit, self.generalist_points, self.bounds)

        delta_points = np.random.uniform(self.bounds[0]/100.0, self.bounds[1]/100.0,
                                         size=(len(self.specialist_points), self.dim))
        child.specialist_points = self.specialist_points + delta_points
        return child

    def crossover(self, other):
        child = KNNBenchmarkEnv(self.dim, self.specialist_fit, self.generalist_fit, self.generalist_points, self.bounds)

        child.specialist_points = 0.5 * (self.specialist_points + other.specialist_points)
        return child

    def __getstate__(self):
        dic = dict()
        dic["dim"] = self.dim
        dic["bounds"] = self.bounds
        dic["gen_points"] = self.generalist_points.tolist()
        dic["gen_fit"] = self.generalist_fit

        dic["spec_points"] = self.specialist_points.tolist()
        dic["spec_fit"] = self.specialist_fit
        return dic

    def __setstate__(self, dic):
        self.__init__(dic["dim"], dic["spec_fit"], dic["gen_fit"], np.array(dic["gen_points"]), dic["bounds"])

        self.specialist_points = np.array(dic["spec_points"])


class KNNBenchmarkEnvFactory(EnvironmentFactory):
    def __init__(self, dimension, spec_fitness, general_fitness, bounds):
        self.dim = dimension
        self.spec_fit = spec_fitness
        self.gen_fit = general_fitness
        self.bounds = bounds

        self.generalist_points = np.random.uniform(bounds[0], bounds[1], size=(len(general_fitness), dimension))

    def new(self):
        return KNNBenchmarkEnv(self.dim, self.spec_fit, self.gen_fit, self.generalist_points, self.bounds)


class KNNBenchmarkAg(Agent):
    def __init__(self, dimension, bounds):
        self.value = np.random.uniform(bounds[0], bounds[1], size=dimension)
        self.bounds = bounds
        self.dim = dimension
        self.opt_state = Configuration.optimizer.default_state()

    def choose_action(self, state):
        return self.value

    def randomize(self):
        self.value = np.random.uniform(self.bounds[0], self.bounds[1], size=self.dim)

    def get_weights(self):
        return self.value

    def set_weights(self, weights):
        self.value = weights

    def get_opt_state(self):
        return self.opt_state

    def set_opt_state(self, state):
        self.opt_state = state

    def __getstate__(self):
        dic = dict()
        dic["value"] = self.value.tolist()
        dic["dim"] = self.dim
        dic["bounds"] = self.bounds
        dic["opt"] = self.opt_state
        return dic

    def __setstate__(self, state):
        self.__init__(state["dim"], state["bounds"])
        self.value = np.array(state["value"])
        self.opt_state = state["opt"]


class KNNBenchmarkAgFactory(AgentFactory):
    def __init__(self, dimension, bounds):
        self.dim = dimension
        self.bounds = bounds

    def new(self):
        return KNNBenchmarkAg(self.dim, self.bounds)


def print_points(points, finess):
    for p in points:
        plt.plot(p[0], p[1], "ob")
    size = 100
    k = np.linspace(-1, 1, num=size)
    a = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            fit = 0
            dist = 0
            for p in range(len(points)):
                d_inv = (np.linalg.norm(points[p] - np.array([k[i], k[j]])) + 1e-9)**-3
                fit += d_inv * finess[p]
                dist += d_inv
            fit /= dist
            a[j, i] = fit
    plt.imshow(a, cmap='hot', interpolation='nearest', extent=[-1, 1, 1, -1])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    finess = np.random.uniform(100, 500, size=20)
    points = np.random.uniform(-1, 1, size=(len(finess), 2))
    plt.show()
    for k in range(100):
        plt.clf()
        plt.title(k)
        print_points(points, finess)
        plt.pause(0.01)
        points += np.random.uniform(-0.1, 0.1, size=(len(finess), 2))