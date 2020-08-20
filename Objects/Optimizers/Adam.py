from ABC.Optimizers import Optimizer
import numpy as np


class Adam(Optimizer):
    def __init__(self, lr_init=0.01, lr_decay=0.9999, lr_limit=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.lr_init = lr_init
        self.lr_decay = lr_decay
        self.lr_limit = lr_limit

    def step(self, gradient, state):
        if state is None:
            state = self.default_state()
            print("yo")

        t = state["t"]
        m = state["m"]
        v = state["v"]

        stepsize = max(self.lr_init * self.lr_decay ** t, self.lr_limit)

        if m is None:
            m = np.zeros(len(gradient), dtype=np.float32)
        if v is None:
            v = np.zeros(len(gradient), dtype=np.float32)

        a = stepsize * np.sqrt(1 - self.beta2 ** t) / (1 - self.beta1 ** t)
        m = self.beta1 * m + (1 - self.beta1) * gradient
        v = self.beta2 * v + (1 - self.beta2) * (gradient * gradient)
        step = -a * m / (np.sqrt(v) + self.epsilon)

        new_state = dict()
        new_state["t"] = t+1
        new_state["m"] = m
        new_state["v"] = v

        return step, new_state

    def default_state(self):
        state = dict()
        state["t"] = 1
        state["m"] = None
        state["v"] = None
        return state
