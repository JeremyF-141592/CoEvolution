class Configuration:
    """
    Make use of differed imports to centralize configuration easily.
    From any script, it is enough to just import Configuration and call its variables when needed, as long as
    .make() was executed at the beginning of the program.
    """
    baseEnv = None
    envInit = None
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
        from Environments.cppn import CppnEnvParams
        from Environments.bipedal_walker_cppn import BipedalWalkerCPPN
        from Agents.NumpyAgent import NeuralAgentNumpyFactory
        from Optimizers.Adam import Adam
        import Utils.Metrics

        # Configuration.agentFactory = NeuralAgentNumpyFactory(24, 4, 2, 20)
        #
        # Configuration.baseEnv = BipedalWalkerCPPN
        # Configuration.envInit = CppnEnvParams()

        Configuration.metric = Utils.Metrics.fitness_bc

        Configuration.optimizer = Adam()

        # ----------------------------------------------------------------
        from Utils.Benchmark import BenchmarkFactory, Benchmark, diag_gaussian

        Configuration.agentFactory = BenchmarkFactory()
        Configuration.benchmark = diag_gaussian
        Configuration.baseEnv = Benchmark

        Configuration.envInit = 14

        # ----------------------------------------------------------------


