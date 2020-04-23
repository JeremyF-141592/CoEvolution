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
    budget_spent = []

    @staticmethod
    def make():
        """Edit this part to easily change configuration."""
        from Environments.CPPN_NEAT import CPPN_NEAT
        from Environments.bipedal_walker_cppn import BipedalWalkerCPPN
        from Agents.NumpyAgent import NeuralAgentNumpyFactory
        import Utils.Metrics
        import Utils.Observers

        Configuration.baseEnv = BipedalWalkerCPPN
        Configuration.flatConfig = CPPN_NEAT()

        Configuration.agentFactory = NeuralAgentNumpyFactory(24, 4, 2, 20)

        Configuration.observer = Utils.Observers.empty_observer
        Configuration.metric = Utils.Metrics.fitness_metric
