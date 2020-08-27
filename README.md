# CoEvolution

This repository aims to implement multiple co-evolution algorithms designed for simulated reinforcement learning.

All work is one as part of an Internship at l'*Institut des Systèmes Intelligents et de Robotique* (ISIR), Sorbonne Université, Paris. 
The main objective of the related internship is to develop a new algorithm based on Quality-Diversity, co-evolving agents and environnements in order to find policies that generalize better.

Key-words : Reinforcement Learning, Co-Evolution, Evolution algorithms, Quality-Diversity


# Implemented algorithms

Currently, the repository includes the full implementations of:

* NSGA-II (Deb, K. et al 2002)
* POET Enhanced (Wang, R. et al. 2020)
* A new NSGA-II inspired co-evolution algorithm

As well as a structure for co-evolution built up in a test/learner fashion:
* IPCA structure (De Jong, E. D. 2004)

However, no algorithm was currently developped according to this structure.

More details can be found in the "Algorithms" section of the wiki.
 
 # Dependencies
 
 **Python 3.6 is required**, due to the frequent use of f-strings. The package f2format (https://github.com/pybpc/f2format) may come in handy.
 
 Main package dependencies are as follow :
 * Numpy / Matplotlib / Scipy
 * Ipyparallel (https://github.com/ipython/ipyparallel)
 * Gym (https://github.com/openai/gym)
 
 Sci-kit Learn and Keras also appear in the code, although they are not used by default.
 
Some environments in the repository also need they own packages, including but not limited to : 
 * Neat-python (https://github.com/CodeReclaimers/neat-python)
 * PyFastSim (https://github.com/alexendy/pyfastsim)
 
 Details can be found the "Environments" section of the wiki.
 
  # Quick-start
  
  The main configuration file, **Parameters.py** uses differed imports to synchronise parts of the code and to make modulation easier. The main parts that may need to be changed often are environments, agents and optimizers. Theses three can be anything inherited from abstract classes defined in ./ABC, defaults are :
  * gym BipedalWalkerV2 with CPPN-drawn landscapes
  * Numpy fully-connected NN with tanh activation
  * Adam optimizer
 
  In order to run any of the main algorithms, one needs to start an ipyparallel cluster beforehand, which needs to be accessible in the same folder as the file that needs to run. As an exemple, running POET Enhanced with default arguments and a local cluster of 32 process :
  > ipcluster start -n 32
  
  > python POET_Main.py
  
  Arguments can be written directly in the shell (--*arg* %d), arguments informations can be displayed with --h or --help.
  
  It is possible to resume any execution with the argument --resume_from *\*folder\**, loading the last indexed Iteration_%d file, archive if needed and the file *Hyperparameters.json* containing previous execution arguments.
