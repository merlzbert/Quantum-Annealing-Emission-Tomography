# Quantum Computing

This repository contains experiments and notebooks on different quantum computers. The notebooks are sorted in a folder designated for the specific quantum computer. At the moment we are mostly investigating noisy ill-posed inverse problems. Therefore, one of the basic applications to implement is matrix inversion.  

# DWave

D-Wave Systems Inc. is a Canadian quantum computing company. D-Wave was the world's first company to sell computers to exploit quantum effects in their operation. In 2020, D-Wave introduced a 5000 qubit system, using their new Pegasus chip with 15 connections per qubit. D-Wave does not implement a generic quantum computer; instead, their computers implement specialized quantum annealing. However, D-Wave announced plans in 2021 that they will work on universal gate-base quantum computers as well in the future. As quantum annealing is naturally restricted to binary quadratic models, DWave released a performance update in 2021 utilizing both quantum and classical computers to allow computations with binary, discrete, integer and real valued variables. These new constrained quadratic models run on designated hybrid solvers.

- [DWave Matrix Inversion](DWave/matrix_inversion_cqm.ipynb)

# IBM

IBM is on the forefront of developing universal quantum computers using superconducting qubits in a gate-based model. Although, the applicability of current quantum computers is limited due to the number of qubits and errors induced, promising algorithms have been proposed once the hardware evolves. For example, Harrow, Hassidim and Lloyd, introduced the HHL algorithm for solving sets of linear equations.

- [IBM Matrix Inversion](IBM/matrixinversion_hhl.ipynb)

