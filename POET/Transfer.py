import numpy as np
from POET.LocalTraining import ES_Step
from Parameters import Configuration


def Evaluate_Candidates(ea_list, env, args):
    c = list()
    for m in range(len(ea_list)):
        c.append(ea_list[m][1])
        c.append(ES_Step(ea_list[m][1], env, args))

    scores = np.zeros(len(c))
    for i in range(args.nb_rounds):
        scores += Configuration.lview.map(env, c)
        Configuration.budget_spent[-1] += len(c)
    scores /= args.nb_rounds
    scores = np.array(scores)
    return c[scores.argmax()]
