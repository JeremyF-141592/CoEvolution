import numpy as np
from Parameters import Configuration
from Templates.Agents import Agent, AgentFactory
from Templates.Environments import Environment, EnvironmentFactory
import matplotlib.pyplot as plt
import os


class KNNBenchmarkEnv(Environment):
    """
    This environment is a N-dimensional space in which some reference points are uniformly distributed with a
    predefined fitness.
    Any agent in this environment is a N-dimensional point, which fitness is the total sum of every reference point
    fitness pondered by the inverse squared distance to the agent.

    A good analogy in 3D is gravity, think each reference point as a planet and an agent as an astronaut,
    its fitness is the total magnitude of the forces that applies to him.

    (Note that this analogy only holds in 3D because in N dimensions each reference fitness should be multiplied
    by the -(N-1) power of distance to conserve energy.)
    """
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

        delta_points = np.random.uniform(self.bounds[0]/20.0, self.bounds[1]/20.0,
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
    def __init__(self, dimension, spec_fitness, general_fitness, bounds, gen_path=""):
        self.dim = dimension
        self.spec_fit = spec_fitness
        self.gen_fit = general_fitness
        self.bounds = bounds

        if gen_path == "":
            path = f"./generalist_benchmark{dimension}.npy"
        else:
            path = gen_path

        if os.path.exists(path):
            self.generalist_points = np.load(path)
        else:
            self.generalist_points = np.random.uniform(bounds[0], bounds[1], size=(len(general_fitness), dimension))
            np.save(path, self.generalist_points)

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

    def __str__(self):
        return str(self.value)


class KNNBenchmarkAgFactory(AgentFactory):
    def __init__(self, dimension, bounds):
        self.dim = dimension
        self.bounds = bounds

    def new(self):
        return KNNBenchmarkAg(self.dim, self.bounds)


def print_points(points, finess, bounds, cut=-1):
    if cut == -1:
        cut = len(points)
    for i in range(cut):
        plt.plot(points[i][0], points[i][1], "ob")
    size = 100
    k = np.linspace(bounds[0], bounds[1], num=size)
    a = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            fit = 0
            dist = 0
            for p in range(len(points)):
                d_inv = (np.linalg.norm(points[p] - np.array([k[i], k[j]])) + 1e-9)**-2
                fit += d_inv * finess[p]
                dist += d_inv
            fit /= dist
            a[j, i] = fit
    plt.imshow(a, cmap='hot', interpolation='nearest', extent=[bounds[0], bounds[1], bounds[1], bounds[0]])


if __name__ == "__main__":
    finess = [0 for i in range(15)] + [10, 30, 50, 70, 90]
    points = np.random.uniform(-1, 1, size=(len(finess), 2))
    plt.show()
    for k in range(100):
        plt.clf()
        plt.title(k)
        print_points(points, finess, [-2, 2])
        plt.pause(0.01)
        points += np.random.uniform(-0.1, 0.1, size=(len(finess), 2))