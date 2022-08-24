# pso_rff
Particle Swarm Optimization for Reactive Force Fields

Dependencies:
- LAMMPS built with ReaxFF 
- First-principles ground truth data

This code is a synthesis of a pre-existing force field optimization code from Matt Curnan [RFFOpt](https://github.com/15garzab/RFFopt) and the [pyswarm](https://github.com/tisimst/pyswarm) package for evolutionary optimization. It performs PSO to find optimal ReaxFF parameters for a chosen chemical system, varying the selected free parameters in the target potential to best fit a selected set of training data plus structures.

/optimize/ contains an example of a working pso_rff run, with included batch script for submission via SLURM. 'help.txt' contains more instructions.

You can test the modular components of the optimization script 'optimize.py' by importing modules from /pso_rff/.
