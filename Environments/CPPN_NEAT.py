import numpy as np
import matplotlib.pyplot as plt


def poly_mut(eta):
    u = np.random.random()
    if u <= 0.5:
        return (2 * u) ** (1 / (1 + eta)) - 1
    else:
        return 1 - (2 * (1 - u)) ** (1 / (1 + eta))


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

    def draw(self, arr, bounds=(-1, 1)):
        res = np.empty(len(arr))
        for i in range(len(arr)):
            res[i] = min(bounds[1], max(bounds[0], self.compute(arr[i])))
        return res

    def draw_noscale(self, arr):
        res = np.empty(len(arr))
        for i in range(len(arr)):
            res[i] = self.compute(arr[i])
        return res

    def usual_poly_mut(self):
        new_genes = list()
        for gene in self.connect_genes:
            if np.random.random() < 2/len(self.connect_genes):
                plmu = poly_mut(4)
                new_gene = (gene[0], gene[1], gene[2] + plmu, gene[3])
                # new_gene = (gene[0], gene[1], gene[2] + 0.5*np.random.random(), gene[3])
                new_genes.append(new_gene)
            else:
                new_genes.append(gene)
        self.connect_genes = new_genes
        self.make()

    def copy(self):
        new_one = CPPN_NEAT()
        new_one.node_genes = self.node_genes.copy()
        new_one.connect_genes = self.connect_genes.copy()
        new_one.hierarchy = self.hierarchy.copy()
        new_one.generation = self.generation
        new_one.make()
        return new_one

    def get_child(self):
        return self.mutate()

    def mutate(self):
        child = self.copy()
        child.generation = self.generation + 1
        # Generate new nodes
        if np.random.random() < 0.8**(len(self.node_genes) - 2):
            # print("NEW NODE")
            child.expand_node()

        # Generate new connections
        if len(child.node_genes) > 3:
            # print("NEW CONNECTION")
            if np.random.random() < 0.8**(len(self.node_genes) - 3):
                child.new_link()

        child.usual_poly_mut()
        return child

    def expand_node(self):
        self.node_genes.append(np.random.randint(0, len(self.map_funcs)))
        chosen_one = np.random.randint(0, len(self.connect_genes))
        new_connections = []
        for i in range(len(self.connect_genes)):
            gene = self.connect_genes[i]
            if i == chosen_one:
                new_connections.append((gene[0], len(self.node_genes) - 1, 0, len(self.connect_genes) - 1))
                new_connections.append((len(self.node_genes) - 1, gene[1], 0, len(self.connect_genes)))
                self.hierarchy.append((self.hierarchy[gene[0]] + self.hierarchy[gene[1]]) / 2)
            else:
                new_connections.append(gene)
        self.connect_genes = new_connections.copy()

    def new_link(self):
        available_pairs = []
        for i in range(len(self.node_genes)):
            for j in range(len(self.node_genes)):
                if self.hierarchy[i] < self.hierarchy[j] - 0.000001:
                    available_pairs.append((i, j))

        for gene in self.connect_genes:
            available_pairs.remove((gene[0], gene[1]))
        if (0, 1) in available_pairs:
            available_pairs.remove((0, 1))

        if len(available_pairs) > 0:
            choice = np.random.randint(0, len(available_pairs))
            one, two = available_pairs[choice]
            self.connect_genes.append((one, two, 0, len(self.connect_genes) - 1))

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
    tests = [CPPN_NEAT() for i in range(8)]
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
        plt.pause(1)
    # test = CPPN_NEAT()
    # for i in range(8):
    #     test = test.get_child()
    #     test.print()

