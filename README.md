# CoEvolution

This repository includes python re-implementation of Evolutionary algorithms such as :
 
 * POET Enhanced (Wang, R. et al. 2020)
 * NSGA-II (Deb, K. et al 2002)
 * IPCA structure (De Jong, E. D. 2004)
 
 All work is one as part of an Internship at l'*Institut des Systèmes Intelligents et de Robotique* (ISIR), Sorbonne Université, Paris. 
 
 # Dependencies
 
 Main package dependencies are as follow :
 * Numpy
 * Ipyparallel (https://github.com/ipython/ipyparallel)
 * Gym (https://github.com/openai/gym)
 * Neat-python (https://github.com/CodeReclaimers/neat-python)
 * Argparse
 
 Sci-kit Learn and Keras also appear in the code, although they are not used by default.
 
  # Quick-start
  
  The main configuration file, **Parameters.py** uses differed imports to synchronise parts of the code and to make modulation easier. The main parts that may need to be changed often are environments, agents and optimizers. Theses three can be anything inherited from abstract classes defined in ./Templates, defaults are :
  * gym BipedalWalkerV2 with CPPN-drawn landscapes
  * Numpy fully-connected NN with tanh activation
  * Adam optimizer
 
  In order to run any of the main algorithms, one needs to start an ipyparallel cluster beforehand, which needs to be accessible in the same folder as the file that needs to run. As an exemple, running POET Enhanced with default arguments and a local cluster of 32 process :
  > ipcluster start -n 32
  
  > python POET_Main.py
  
  Arguments can be written directly in the shell (--*arg* %d), and are detailed in each algorithm's main file.
  
  It is possible to resume any execution with the argument --resume *\*folder\**, loading the last indexed Iteration_%d file, archive if needed and the file *commandline_args.txt* containing previous execution arguments.
