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

    # Ipyparallel
    lview = None
    rc = None

    # Execution's trace
    archive = []
    budget_spent = []

    @staticmethod
    def make():
        """Edit this part to easily change configuration."""
        from Objects.Optimizers.Adam import Adam
        import Utils.Metrics

        Configuration.metric = Utils.Metrics.fitness_bc

        Configuration.optimizer = Adam()

        # ----------------------------------------------------------------
        #   CollectBall
        # ----------------------------------------------------------------
        from Objects.Environments.CollectBall import CollectBallFactory
        from Objects.Agents.NumpyAgent import NeuralAgentNumpyFactory

        Configuration.agentFactory = NeuralAgentNumpyFactory(n_in=18,
                                                             n_out=3,
                                                             n_hidden_layers=2,
                                                             n_neurons_per_hidden=10)
        Configuration.envFactory = CollectBallFactory(nb_balls=8,
                                                      setup="SetupMedium.xml")

        """
        # ----------------------------------------------------------------
        #   BipedalWalker
        # ----------------------------------------------------------------
        from Objects.Environments.BipedalWalkerCppn import BipedalWalkerFactory
        from Objects.Agents.NumpyAgent import NeuralAgentNumpyFactory

        Configuration.agentFactory = NeuralAgentNumpyFactory(24, 4, 2, 20)
        Configuration.envFactory = BipedalWalkerFactory()
        """
        """
        # ----------------------------------------------------------------
        #   Benchmark
        # ----------------------------------------------------------------
        from Objects.Environments.DistanceBenchmark import DistanceBenchmarkAgFactory, DistanceBenchmarkEnvFactory

        Configuration.envFactory = DistanceBenchmarkEnvFactory(
            60,
            [0 for i in range(40)] + [10 * i%10 for i in range(20)],
            [10 * i%9 + 5 for i in range(18)],
            [-1, 1]
        )
        Configuration.agentFactory = DistanceBenchmarkAgFactory(
            60,
            [-1, 1]
        )
        """
