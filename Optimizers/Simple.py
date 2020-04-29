from Utils.Optimizers import Optimizer


class Simple(Optimizer):
    def step(self, gradient, state, args):
        t = state

        alpha = max(args.lr_decay ** t, args.lr_limit)
        step = gradient * alpha

        return step, t+1

    def default_state(self):
        return 0
