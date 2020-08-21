from ABC.Optimizers import Optimizer


class Simple(Optimizer):
    def __init__(self, lr_init=0.01, lr_decay=0.9999, lr_limit=0.001):
        self.lr_init = lr_init
        self.lr_decay = lr_decay
        self.lr_limit = lr_limit

    def step(self, gradient, state):
        t = state

        alpha = max(self.lr_init * self.lr_decay ** t, self.lr_limit)
        step = gradient * alpha

        new_state = dict()
        new_state["t"] = t+1
        return step, new_state

    def default_state(self):
        state = dict()
        state["t"] = 1
        return state
