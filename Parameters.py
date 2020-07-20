class Configuration:
    """
    Make use of differed imports to centralize configuration easily.
    From any script, it is enough to just import Configuration and call its variables when needed, as long as
    .make() was executed at the beginning of the program.
    """
    envFactory = None
    agentFactory = None
    metric = None
    optimizer = None

    lview = None
    rc = None

    archive = []
    budget_spent = []

    benchmark = None

    @staticmethod
    def make():
        """Edit this part to easily change configuration."""
        from Optimizers.Adam import Adam
        import Utils.Metrics

        Configuration.metric = Utils.Metrics.fitness_bc

        Configuration.optimizer = Adam()

        # ----------------------------------------------------------------
        #   CollectBall
        # ----------------------------------------------------------------
        from Environments.CollectBall import CollectBallFactory
        from Agents.NumpyAgent import NeuralAgentNumpyFactory

        Configuration.agentFactory = NeuralAgentNumpyFactory(7, 3, 2, 20)
        Configuration.envFactory = CollectBallFactory()

        """
        # ----------------------------------------------------------------
        #   BipedalWalker
        # ----------------------------------------------------------------
        from Environments.bipedal_walker_cppn import BipedalWalkerFactory
        from Agents.NumpyAgent import NeuralAgentNumpyFactory

        Configuration.agentFactory = NeuralAgentNumpyFactory(24, 4, 2, 20)
        Configuration.envFactory = BipedalWalkerFactory()
        """

        """
        # ----------------------------------------------------------------
        #   Benchmark
        # ----------------------------------------------------------------
        from Environments.KNN_Benchmark import KNNBenchmarkAgFactory, KNNBenchmarkEnvFactory

        Configuration.envFactory = KNNBenchmarkEnvFactory(
            2,
            [0 for i in range(15)],
            [20, 40, 60, 80],
            [-1, 1]
        )
        Configuration.agentFactory = KNNBenchmarkAgFactory(
            2,
            [-1, 1]
        )
        """


