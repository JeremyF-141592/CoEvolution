class Configuration:
    """
    Make use of differed imports to centralize configuration easily.
    From any script, it is enough to just import Configuration and call its variables when needed, as long as
    .make() was executed at the beginning of the program.
    """
    baseEnv = None
    flatConfig = None
    agentFactory = None
    observer = None
    metric = None
    optimizer = None
    lview = None
    rc = None
    archive = []
    opt_states = []
    budget_spent = []
    knn = 5

    @staticmethod
    def make():
        """Edit this part to easily change configuration."""
        from Environments.bipedal_walker_custom import BipedalWalkerCustom, Env_config
        from Agents.NumpyAgent import NeuralAgentNumpyFactory
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
        Configuration.agentFactory = NeuralAgentNumpyFactory(24, 4, 2, 10)

        Configuration.observer = Utils.Observers.empty_observer
        Configuration.metric = Utils.Metrics.fitness_metric
