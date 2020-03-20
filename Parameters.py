from Environments.bipedal_walker_custom import BipedalWalkerCustom, Env_config
from Agents.KerasAgent import NeuralAgentFactory

baseEnv = BipedalWalkerCustom
flatConfig = Env_config(name='flat',
                        ground_roughness=0,
                        pit_gap=[],
                        stump_width=[],
                        stump_height=[],
                        stump_float=[],
                        stair_height=[],
                        stair_width=[],
                        stair_steps=[])
agentFactory = NeuralAgentFactory([24, 12, 4])
