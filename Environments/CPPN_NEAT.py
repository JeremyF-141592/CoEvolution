import numpy as np
import matplotlib.pyplot as plt


class CPPN_Node:
    def __init__(self, func):
        self.func = func
        self.in_list = []

    def compute(self, x):
        if len(self.in_list) == 0:
            return x
        res = 0
        for i in range(len(self.in_list)):
            res += self.in_list[i]
        return self.func(res)

    def refresh(self):
        self.in_list = []


class CPPN_NEAT:
    def __init__(self):
        self.node_genes = [0, 0]  # nb 0 is always input, nb 1 is always output
        self.connect_genes = [(0, 1, 0, 0)]  # In - out - weight - innovation- enabled

        self.hierarchy = [0.0, 1.0]
        self.generation = 0

        self.nodes = []
        # Represent functions as integers for easy pickling
        self.map_funcs = [
            CPPN_NEAT.id,
            CPPN_NEAT.sinus,
            CPPN_NEAT.triangles,
            CPPN_NEAT.gaussian,
            CPPN_NEAT.sym,
            CPPN_NEAT.sigmoid,
            # CPPN_NEAT.half_id,
            CPPN_NEAT.bump,
            CPPN_NEAT.fast,
            CPPN_NEAT.slow,
            # CPPN_NEAT.noise,
            np.tanh
        ]
        self.make()

    def make(self):
        self.nodes = []
        for gene in self.node_genes:
            self.nodes.append(CPPN_Node(self.map_funcs[gene]))

    def compute(self, x):
        expected = [0 for i in range(len(self.nodes))]
        for gene in self.connect_genes:
            expected[gene[1]] += 1
        self.connect_genes.sort(key=lambda k: self.hierarchy[k[0]])
        for gene in self.connect_genes:
            self.nodes[gene[1]].in_list.append(self.nodes[gene[0]].compute(x) * gene[2])
        result = self.nodes[1].compute(x)
        for node in self.nodes:
            node.refresh()
        return result

    def draw(self, arr, scale=(0, 1)):
        res = np.empty(len(arr))
        for i in range(len(arr)):
            res[i] = self.compute(arr[i])
        diff = abs(res.max() - res.min() + 0.000001)
        rescale = scale[1] / diff
        res *= rescale

        rescale = res[0] - scale[0]
        res -= rescale
        return res

    def random_step(self, step_size):
        new_connect = list()
        for gene in self.connect_genes:
            new_weight = gene[2] + np.random.normal(0, 1) * step_size
            new_connect.append((gene[0], gene[1], new_weight, gene[3]))
        self.connect_genes = new_connect
        self.make()

    def copy(self):
        new_one = CPPN_NEAT()
        new_one.node_genes = self.node_genes.copy()
        new_one.connect_genes = self.connect_genes.copy()
        new_one.hierarchy = self.hierarchy.copy()
        new_one.generation = self.generation
        new_one.make()
        return new_one

    def __getstate__(self):
        dic = dict()
        dic["node_genes"] = self.node_genes
        dic["connect_genes"] = self.connect_genes
        dic["hierarchy"] = self.hierarchy
        dic["generation"] = self.generation
        return dic

    def __setstate__(self, dic):
        self.__init__()
        self.node_genes = dic["node_genes"]
        self.connect_genes = dic["connect_genes"]
        self.hierarchy = dic["hierarchy"]
        self.generation = dic["generation"]
        self.make()

    def get_child(self):
        return self.mutate()

    def mutate(self):
        child = self.copy()
        child.generation = self.generation + 1
        # Generate new nodes
        if np.random.random() < 0.7**(len(self.node_genes) - 2):
            # print("NEW NODE")
            child.node_genes.append(np.random.randint(0, len(self.map_funcs)))
            chosen_one = np.random.randint(0, len(child.connect_genes))
            new_connections = []
            for i in range(len(child.connect_genes)):
                gene = child.connect_genes[i]
                if i == chosen_one:
                    new_connections.append((gene[0], len(child.node_genes)-1, 0, len(child.connect_genes)-1))
                    new_connections.append((len(child.node_genes)-1, gene[1], 0, len(child.connect_genes)))
                    child.hierarchy.append((child.hierarchy[gene[0]] + child.hierarchy[gene[1]]) / 2)
                else:
                    new_connections.append(gene)
            child.connect_genes = new_connections.copy()
        # Generate new connections
        if len(child.node_genes) > 3:
            # print("NEW CONNECTION")
            if np.random.random() < 0.7 ** (len(self.node_genes) - 3):
                available_pairs = []
                for i in range(len(child.node_genes)):
                    for j in range(len(child.node_genes)):
                        if child.hierarchy[i] < child.hierarchy[j]-0.000001:
                            available_pairs.append((i, j))

                for gene in child.connect_genes:
                    available_pairs.remove((gene[0], gene[1]))
                if (0, 1) in available_pairs:
                    available_pairs.remove((0, 1))

                if len(available_pairs) > 0:
                    choice = np.random.randint(0, len(available_pairs))
                    one, two = available_pairs[choice]
                    child.connect_genes.append((one, two, 0, len(child.connect_genes)-1))

        child.random_step(0.2)
        child.make()
        return child

    def print(self):
        print("Generation :", self.generation)
        print("Nodes :", self.node_genes)
        print("Hierarchy :", self.hierarchy)
        print("Connections : \n", self.connect_genes, "\n")

    @staticmethod
    def reproduce(a, b):
        pass

    @staticmethod
    def randomize():
        child = CPPN_NEAT()
        max_mut = np.random.randint(1, 14)
        for i in range(max_mut):
            child = child.mutate()
        child.random_step(0.5)
        return child

    @staticmethod
    def sym(x):
        return -x + 1 if x > 0 else x + 1

    @staticmethod
    def sigmoid(x):
        return 2. / (1 + np.exp(-x))

    @staticmethod
    def gaussian(x):
        return 4 * np.exp(-((x / 4) ** 2))

    @staticmethod
    def sinus(x):
        return np.sin(x / 2)

    @staticmethod
    def triangles(x):
        return x % 1

    @staticmethod
    def id(x):
        return x

    @staticmethod
    def noise(x):
        return np.random.random()

    @staticmethod
    def half_id(x):
        return x if x > 0 else 0

    @staticmethod
    def bump(x):
        return np.sqrt(1 - x ** 2) if -1 < x < 1 else 0

    @staticmethod
    def fast(x):
        return 10*x

    @staticmethod
    def slow(x):
        return x/5


if __name__ == "__main__":
    tests = [CPPN_NEAT() for i in range(12)]
    lin = np.linspace(-10, 10, num=100)
    plt.show()
    for i in range(20):
        plt.clf()
        plt.xlim(-6, 6)
        plt.ylim(-8.2, 8.2)
        plt.title(f"iteration {i}")
        for j in range(len(tests)):
            plt.plot(lin, tests[j].draw(lin, scale=(0, min(8, 1.4*i))), label=f"{j}")
            tests[j] = tests[j].get_child()
            print(f"--- {j} ---")
            tests[j].print()
        plt.legend()
        plt.pause(0.4)
    # test = CPPN_NEAT()
    # for i in range(8):
    #     test = test.get_child()
    #     test.print()

