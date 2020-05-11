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

    nb_rounds = 1

    benchmark = None

    @staticmethod
    def make():
        """Edit this part to easily change configuration."""
        from Environments.cppn import CppnEnvParams
        from Environments.bipedal_walker_cppn import BipedalWalkerCPPN
        from Agents.NumpyAgent import NeuralAgentNumpyFactory
        from Optimizers.Adam import Adam
        import Utils.Metrics
        import Utils.Observers

        Configuration.agentFactory = NeuralAgentNumpyFactory(24, 4, 2, 20)

        Configuration.observer = Utils.Observers.empty_observer
        Configuration.metric = Utils.Metrics.fitness_metric

        Configuration.optimizer = Adam()

        # ----------------------------------------------------------------
        from Utils.Benchmark import gramacy_lee, custom, Benchmark

        Configuration.benchmark = custom
        Configuration.baseEnv = Benchmark

        Configuration.flatConfig = (len(Configuration.agentFactory.new().get_weights()),
                                    1.2,
                                    1.0)

        # ----------------------------------------------------------------

        # Configuration.baseEnv = BipedalWalkerCPPN
        # Configuration.flatConfig = CppnEnvParams()
