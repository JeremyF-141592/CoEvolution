from Utils.Optimizers import Optimizer
import numpy as np


class Adam(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-08):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def step(self, gradient, state, args):
        t, m, v = state
        stepsize = max(args.lr_init * args.lr_decay ** t, args.lr_limit)

        if m is None:
            m = np.zeros(len(gradient), dtype=np.float32)
        if v is None:
            v = np.zeros(len(gradient), dtype=np.float32)

        print("   ", t)
        a = stepsize * np.sqrt(1 - self.beta2 ** t) / (1 - self.beta1 ** t)
        m = self.beta1 * m + (1 - self.beta1) * gradient
        v = self.beta2 * v + (1 - self.beta2) * (gradient * gradient)
        step = -a * m / (np.sqrt(v) + self.epsilon)

        return step, (t+1, m, v)

    def default_state(self):
        return 1, None, None
