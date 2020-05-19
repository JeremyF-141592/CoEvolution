from keras import Sequential
from keras.layers import Dense
import numpy as np
from Templates.Agents import Agent, AgentFactory


class NeuralAgent(Agent):
    """
    Simple dense neural agent.
    """
    def __init__(self, size_list, activation="tanh"):
        """"The shape of the layers are one dimensional and taken as a list, from input size to output size."""
        assert(len(size_list) > 1)
        model = Sequential()
        for i in range(1, len(size_list)-1):
            model.add(Dense(size_list[i], input_shape=(size_list[i-1],)))
        model.add(Dense(size_list[len(size_list)-1], activation=activation))
        model.build(input_shape=(size_list[0],))
        self.model = model
        self.size_list = size_list
        self.opt_state = []

    def randomize(self):
        wei = self.get_weights()
        wei = np.random.uniform(-1, 1, wei.shape)
        self.set_weights(wei)

    def get_weights(self):
        # Return weights as a flattened array
        wei = self.model.get_weights()
        res = np.array([])
        for w in wei:
            res = np.hstack((res, w.flatten()))
        return res

    def set_weights(self, weights):
        # Set weights from a flattened array
        wei = self.model.get_weights()
        res = []
        count = 0
        for w in wei:
            res.append(np.array(weights[count:count + w.size]).reshape(w.shape))
            count += w.size
        self.model.set_weights(res)

    def choose_action(self, state):
        state = np.array([state, ])
        return self.model.predict(state)[0]

    def __str__(self):
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        return "\n".join(stringlist)

    def get_opt_state(self):
        return self.opt_state

    def set_opt_state(self, state):
        self.opt_state = state

    def __getstate__(self):
        dic = dict()
        dic["as_vector"] = self.get_weights()
        dic["size_list"] = self.size_list
        return dic

    def __setstate__(self, state):
        self.__init__(state["size_list"])
        self.set_weights(state["as_vector"])


class NeuralAgentFactory(AgentFactory):
    def __init__(self, size_list, activation="tanh"):
        assert (len(size_list) > 1)
        self.size_list = size_list
        self.activation = activation

    def new(self):
        return NeuralAgent(self.size_list, self.activation)
