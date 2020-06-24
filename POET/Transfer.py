import numpy as np
from POET.LocalTraining import ES_Step
from Parameters import Configuration


def Evaluate_Candidates(ea_list, env, args, threshold=0):
    """As in POET enhanced, evaluation on agents first, followed by validation on fine-tuned agents."""
    base_agents = list()
    for ea_pair in ea_list:
        base_agents.append(ea_pair[1])

    scores = np.zeros(len(base_agents))

    fitness = Configuration.lview.map(env, base_agents)
    scores += fitness
    Configuration.budget_spent[-1] += len(base_agents)

    proposed_agents = list()
    full_scores = list()

    fine_tuned_agents = list()
    for i in range(len(base_agents)):
        if scores[i] > threshold:
            proposed_agents.append(base_agents[i])
            full_scores.append(scores[i])

            theta, _ = ES_Step(base_agents[i], env, args)
            fine_tuned_agents.append(theta)

    if len(fine_tuned_agents) == 0:  # No other agent has a fitness above the threshold
        return None, float("-inf")
    scores = np.zeros(len(fine_tuned_agents))

    fitness = Configuration.lview.map(env, fine_tuned_agents)
    scores += fitness
    Configuration.budget_spent[-1] += len(base_agents)

    for i in range(len(fine_tuned_agents)):
        if scores[i] > threshold:
            proposed_agents.append(fine_tuned_agents[i])
            full_scores.append(scores[i])
    full_scores = np.array(full_scores)

    return proposed_agents[full_scores.argmax()], full_scores.max()
