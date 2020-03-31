class Configuration:
    baseEnv = None
    flatConfig = None
    agentFactory = None
    observer = None
    metric = None
    lview = None
    rc = None

    @staticmethod
    def make():
        from Environments.bipedal_walker_custom import BipedalWalkerCustom, Env_config
        from Agents.KerasAgent import NeuralAgentFactory
        import Utils.Metrics
        import Utils.Observers

        Configuration.baseEnv = BipedalWalkerCustom
        Configuration.flatConfig = Env_config(name='flat',
                                              ground_roughness=0,
                                              pit_gap=[],
                                              stump_width=[],
                                              stump_height=[],
                                              stump_float=[],
                                              stair_height=[],
                                              stair_width=[],
                                              stair_steps=[])
        Configuration.agentFactory = NeuralAgentFactory([24, 12, 4])

        Configuration.observer = Utils.Observers.empty_observer
        Configuration.metric = Utils.Metrics.fitness_metric
