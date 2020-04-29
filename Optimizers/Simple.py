from Utils.Optimizers import Optimizer


class Simple(Optimizer):
    def step(self, gradient, state, args):
        t = state

        alpha = max(args.lr_init * args.lr_decay ** t, args.lr_limit)
        step = gradient * alpha

        new_state = dict()
        new_state["t"] = t+1
        return step, new_state

    def default_state(self):
        state = dict()
        state["t"] = 1
        return state
