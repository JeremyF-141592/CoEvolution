from Environments.bipedal_walker_custom import BipedalWalkerCustom, Env_config
from Parameters import agentFactory

env_c = Env_config(
                name='flat',
                ground_roughness=0,
                pit_gap=[],
                stump_width=[],
                stump_height=[],
                stump_float=[],
                stair_height=[],
                stair_width=[],
                stair_steps=[])

env = BipedalWalkerCustom(env_c)

agent = agentFactory.new()
agent.randomize()

for i in range(20):
    print(env(agent))