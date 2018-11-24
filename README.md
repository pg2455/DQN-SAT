# Reinforcement Learning for SATISFIABILITY Problems
The repository contains the effort to make reinforcement learning models work for Satisfiability solving. This repo contains the code to learn the DQN agent. After a lot of efforts when I couldn't make the agent predict sensible Q-values, I had to discard the project. However, there has been some recent work on the same thing like [SAT Solving in the style of Alpha Go](https://arxiv.org/pdf/1802.05340.pdf), [Learning through Single-bit supervision](https://arxiv.org/pdf/1802.03685.pdf) and [Automated Reasoning for SAT Solvers](https://arxiv.org/abs/1807.08058)

## Python interface to SAT Solvers
Various solvers like [minisat](http://minisat.se/), [glucose-syrup](http://www.labri.fr/perso/lsimon/glucose/) and several others are coded in C++.
It gets tough for someone less familiar with C++ to make these solvers interface with machine learning libraries in other languages.
This repository contains an easier interface that involves reading from and writing to common text files.
There are two text files:
 - variable_selection.txt - This file contains the literal and boolean value to assign to that literal. It forms the instruction for the solver.
 - results.txt - This file contains the output from the solver observed when the instructions from above are executed.

#### How to install minisat solver?
- Download and untar the code repo from the [wesbite](http://minisat.se/MiniSat.html).
- Two files - [Solver.cc](minisat2/core/Solver.cc) and [Solver.h](minisat2/core/Solver.h) require some changes to make the solver read and write to an external file. These changes are denoted by the comment `//codedits`. Specifically, `Solver.h` contains the location of the read and write text files.
- The [README](minisat2/README) file in the repo contains further installation instructions. Above changes have been made with respect to the code in [core](minisat2/core) directory, so the installation need to follow with respect to `core`.

**Note**: When the sovler is run from Python it can't be stopped with `Ctrl+C`, so it is required to find the process id and kill it manually. This bash script might come in handy:
`kill $(ps -Af | grep minisat | awk '{print $2}')`.

## Requirements
Following Python2.7 libraries are required:
- [pytorch](https://pytorch.org/) for deep learning models
- [visdom](https://github.com/facebookresearch/visdom) for visualising the progress
- [logger](https://github.com/oval-group/logger) to easily interface with visdom
- [pympler](https://pythonhosted.org/Pympler/) to watch the memory usage

## Synthetic Data
The data used in learning of the DQN agent requires Satisfiability problems. The `main_data_generation.py` generates such formulas corresponding to the Hamiltonian Cycle/Path decision problem on graphs. The output is the the [CNF](https://en.wikipedia.org/wiki/Conjunctive_normal_form) for each graph. Graphs are randomly generated using one of [Barabasi-Albert model](https://en.wikipedia.org/wiki/Barab%C3%A1si%E2%80%93Albert_model) or [Erdos-Renyi model](https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93R%C3%A9nyi_model).

## DQN Agent
The DQN agent is inspired from [Learning Combinatorial Optimization Algorithms over Graphs](https://arxiv.org/abs/1704.01665). The GNN model is trained on the graph corresponding to [Clause Variable Incidence Graph](http://www.iiia.csic.es/~levy/papers/slidesCCIA11.pdf) conversion.
`Q_network.py` contains the neural architecture to learn the graph embedding and the Q-function.

## How to run the code?
- Make sure that there are empty text files in the repo named `variable_selection.txt` and `results.txt`
- Make desired changes to `configuration.py`
- Generate sythetic data using `main_data_generation.py`
- Let the DQN agent learn using `main_dqn.py`
