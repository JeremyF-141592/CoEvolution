import numpy as np
from Templates.Agents import Agent, AgentFactory
from Parameters import Configuration


def sigmoid(x):
    return 1./(1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)
    

class NeuralAgentNumpy(Agent):
    def __init__(self, n_in, n_out, n_hidden_layers=2, n_neurons_per_hidden=5):
        self.dim_in = n_in
        self.dim_out = n_out
        self.n_per_hidden = n_neurons_per_hidden
        self.n_hidden_layers = n_hidden_layers
        self.weights = None 
        self.n_weights = None
        self.opt_state = Configuration.optimizer.default_state()
        self.randomize()
        self.out = np.zeros(n_out)
        #print("Creating a simple mlp with %d inputs, %d outputs, %d hidden layers and %d neurons per layer"%(n_in, n_out,n_hidden_layers, n_neurons_per_hidden))
    
    def randomize(self):
        if(self.n_hidden_layers > 0):
            self.weights = [2*np.random.random((self.dim_in,self.n_per_hidden))-1] # In -> first hidden
            self.bias = [2*np.random.random(self.n_per_hidden)-1] # In -> first hidden
            for i in range(self.n_hidden_layers-1): # Hidden -> hidden
                self.weights.append(2*np.random.random((self.n_per_hidden,self.n_per_hidden))-1)
                self.bias.append(2*np.random.random(self.n_per_hidden)-1)
            self.weights.append(2*np.random.random((self.n_per_hidden,self.dim_out))-1) # -> last hidden -> out
            self.bias.append(2*np.random.random(self.dim_out)-1)
        else:
            self.weights = [2*np.random.random((self.dim_in,self.dim_out))-1] # Single-layer perceptron
            self.bias = [2*np.random.random(self.dim_out)-1]
        self.n_weights = np.sum([np.product(w.shape) for w in self.weights]) + np.sum([np.product(b.shape) for b in self.bias])

    def get_weights(self):
        """
        Returns all network parameters as a single array
        """
        flat_weights = np.hstack([arr.flatten() for arr in (self.weights+self.bias)])
        return flat_weights

    def set_weights(self, flat_parameters):
        """
        Set all network parameters from a single array
        """
        if (np.nan in flat_parameters):
            print("WARNING: NaN in the parameters of the NN: "+str(list(flat_parameters)))
        if (max(flat_parameters)>1000):
            print("WARNING: max value of the parameters of the NN >1000: "+str(list(flat_parameters)))
            
                
        i = 0 # index
        to_set = []
        self.weights = list()
        self.bias = list()
        if(self.n_hidden_layers > 0):
            # In -> first hidden
            w0 = np.array(flat_parameters[i:(i+self.dim_in*self.n_per_hidden)])
            self.weights.append(w0.reshape(self.dim_in,self.n_per_hidden))
            i += self.dim_in*self.n_per_hidden
            for l in range(self.n_hidden_layers-1): # Hidden -> hidden
                w = np.array(flat_parameters[i:(i+self.n_per_hidden*self.n_per_hidden)])
                self.weights.append(w.reshape((self.n_per_hidden,self.n_per_hidden)))
                i += self.n_per_hidden*self.n_per_hidden
            # -> last hidden -> out
            wN = np.array(flat_parameters[i:(i+self.n_per_hidden*self.dim_out)])
            self.weights.append(wN.reshape((self.n_per_hidden,self.dim_out)))
            i += self.n_per_hidden*self.dim_out
            # Samefor bias now
            # In -> first hidden
            b0 = np.array(flat_parameters[i:(i+self.n_per_hidden)])
            self.bias.append(b0)
            i += self.n_per_hidden
            for l in range(self.n_hidden_layers-1): # Hidden -> hidden
                b = np.array(flat_parameters[i:(i+self.n_per_hidden)])
                self.bias.append(b)
                i += self.n_per_hidden
            # -> last hidden -> out
            bN = np.array(flat_parameters[i:(i+self.dim_out)])
            self.bias.append(bN)
            i += self.dim_out
        else:
            n_w = self.dim_in*self.dim_out
            w = np.array(flat_parameters[:n_w])
            self.weights = [w.reshape((self.dim_in,self.dim_out))]
            self.bias = [np.array(flat_parameters[n_w:])]
        self.n_weights = np.sum([np.product(w.shape) for w in self.weights]) + np.sum([np.product(b.shape) for b in self.bias])
    
    def choose_action(self,x):
        """
        Propagate
        """
        if(self.n_hidden_layers > 0):
            #Input
            y = np.matmul(x,self.weights[0]) + self.bias[0]
            y = tanh(y)
            # hidden -> hidden
            for i in range(1,self.n_hidden_layers-1):
                y = np.matmul(y, self.weights[i]) + self.bias[i]
                y= tanh(y)
            # Out
            a = np.matmul(y, self.weights[-1]) + self.bias[-1]
            out = tanh(a)
            return out
        else: # Simple monolayer perceptron
            return tanh(np.matmul(x,self.weights[0]) + self.bias[0])

    def get_opt_state(self):
        return self.opt_state

    def set_opt_state(self, state):
        self.opt_state = state

    def __getstate__(self):
        dic = dict()
        dic["dim_in"] = self.dim_in
        dic["dim_out"] = self.dim_out
        dic["n_hidden_layers"] = self.n_hidden_layers
        dic["n_per_hidden"] = self.n_per_hidden
        dic["opt_state"] = self.opt_state
        dic["as_vector"] = self.get_weights()
        return dic

    def __setstate__(self, dic):
        self.__init__(dic["dim_in"], dic["dim_out"], dic["n_hidden_layers"], dic["n_per_hidden"])
        self.set_weights(dic["as_vector"])
        self.set_opt_state(dic["opt_state"])


class NeuralAgentNumpyFactory(AgentFactory):
    def __init__(self, n_in, n_out, n_hidden_layers=2, n_neurons_per_hidden=5):
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons_per_hidden = n_neurons_per_hidden

    def new(self):
        return NeuralAgentNumpy(self.n_in, self.n_out, self.n_hidden_layers, self.n_neurons_per_hidden)