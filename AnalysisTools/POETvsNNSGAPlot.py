"""
FILE IS TO BE CLEANED AND REFACTORED

If you can read this warning, chances are the file will be unreadable and not useful.

"""

from Objects.Environments.Benchmark import *
import matplotlib.pyplot as plt
import seaborn as sns

res_POET = np.load("../temp/Results_CollectNNSGA_GL.npy")

res_NNSGA = np.load("../temp/Results_CollectNNSGA_L.npy")

print(res_POET.shape)

print(res_NNSGA.shape)

# Max. fitness (1 env, over agents)
Env_solvability_POET = list()
Env_solvability_NNSGA = list()

# Median fitness (1 ag, over environments)
Ag_generalist_POET = list()
Ag_generalist_NNSGA = list()

for env in res_POET:
    Env_solvability_POET.append(env.max())
for env in res_NNSGA:
    Env_solvability_NNSGA.append(env.max())

for ag in res_POET.T:
    Ag_generalist_POET.append(np.median(ag))
for ag in res_NNSGA.T:
    Ag_generalist_NNSGA.append(np.median(ag))

Ag_generalist_NNSGA = np.array(Ag_generalist_NNSGA)
best_o = res_NNSGA.T[Ag_generalist_NNSGA.argsort()[-1]]
print(np.median(best_o))
print(best_o.mean())
print(best_o.std())

Ag_generalist_POET = np.array(Ag_generalist_POET)
best_o = res_POET.T[Ag_generalist_POET.argsort()[-1]]
print(np.median(best_o))
print(best_o.mean())
print(best_o.std())

sns.set_style('whitegrid')
plt.title("Solvability score distribution for each environment - 10 run - 2D-NN Benchmark")
plt.xlabel("Maximum Fitness")
plt.ylabel("Environment density")
sns.kdeplot(np.array(Env_solvability_POET), color="deepskyblue", label="POET")
# plt.hist(Env_solvability_POET, bins=50, alpha=0.5)
plt.axvline(np.median(Env_solvability_POET), linestyle="--", color="deepskyblue")

sns.kdeplot(np.array(Env_solvability_NNSGA), color="orange", label="NNSGA")
# plt.hist(Env_solvability_NNSGA, bins=50, alpha=0.5)
plt.axvline(np.median(Env_solvability_NNSGA), linestyle="--", color="orange")
plt.show()
plt.title("Generalisation score distribution for each agent - 10 run - 2D-NN Benchmark ")
plt.xlabel("Median Fitness")
plt.ylabel("Proportion of agents")
sns.kdeplot(np.array(Ag_generalist_POET), color="deepskyblue", label=f"POET - {len(Ag_generalist_POET)} agents")
plt.axvline(np.median(Ag_generalist_POET), linestyle="--", color="deepskyblue")

sns.kdeplot(np.array(Ag_generalist_NNSGA), color="orange", label=f"NNSGA {len(Ag_generalist_NNSGA)} agents")
plt.axvline(np.median(Ag_generalist_NNSGA), linestyle="--", color="orange")
plt.show()

# fig1, ax1 = plt.subplots()
# ax1.set_title('Random environment test over 5 independent benchmark runs')
# ax1.boxplot([Env_solvability_POET, Env_solvability_NNSGA, Ag_generalist_POET, Ag_generalist_NNSGA], whis=float("inf"))
#
# fig1.canvas.draw()
#
# labels = [item.get_text() for item in ax1.get_xticklabels()]
# labels[0] = 'Solvability POET'
# labels[1] = 'Solvability NNSGA'
# labels[2] = 'Generalization POET'
# labels[3] = 'Generalization NNSGA'
#
#
# ax1.set_xticklabels(labels)
# plt.ylabel("Fitness")
# plt.show()