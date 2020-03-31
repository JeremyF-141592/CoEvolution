import numpy as np
from POET.LocalTraining import ES_Step
from Parameters import Configuration


def Evaluate_Candidates(ea_list, env, args):
    c = list()
    for m in range(len(ea_list)):
        c.append(ea_list[m][1])
        c.append(ES_Step(ea_list[m][1], env, args))

    scores = Configuration.lview.map(env, c)
    scores = np.array(scores)
    return c[scores.argmax()]
