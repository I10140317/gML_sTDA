# gML-sTDA

## Overview

**gML-sTDA** is a machine-learning accelerated implementation of the **simplified Tamm–Dancoff approximation (sTDA)** for excited-state calculations.

The method integrates a **generative Restricted Boltzmann Machine (RBM)** with the sTDA framework to efficiently explore the configuration space of electronic excitations. By learning the statistical distribution of important configurations, gML-sTDA significantly reduces the effective configuration space and accelerates excited-state calculations.

This approach enables efficient simulations of excited states for large molecular systems while maintaining good accuracy.

---

## Method

The key idea of **gML-sTDA** is to use a **generative RBM model** to identify and sample the most relevant excited-state configurations.

Compared with conventional sTDA calculations, this approach:

- reduces the effective configuration space
- accelerates Hamiltonian construction
- lowers computational cost
- enables calculations for larger systems

The general workflow of gML-sTDA includes:

1. Generation of AO integrals
2. RBM-based sampling of important configurations
3. Construction of the reduced sTDA Hamiltonian
4. Diagonalization to obtain excited-state energies

---

## Requirements

Typical dependencies include:

- python==3.12
- numba==0.60.0
- numpy==1.26.4
- scipy==1.14.1
- pyscf==2.11.1
- setuptools==72.1.0
- torch==2.5.1
- fortran compilers (recommanded ifx/ifort) to compile modified version of stda

Install dependencies using:
pip install -r requirements.txt

## Installation
git clone https://github.com/I10140317/gML-sTDA.git

cd gML-sTDA

./install.sh

## Usage
A typical workflow consists of:
1. Preparing a molden file
2. Moldify parameters in the select_config.py
3. Solving for excited states

Example:

cd example

python3 select_config.py > out.log

## Output
The program provides:
1. Excited-state energies
2. CIS coefficients
3. Additional analysis for excited-state properties

## License
This project includes components derived from the original sTDA implementation developed by the Grimme group, which is licensed under GPL-3.0.

All modified components remain distributed under the same license.

## Citation
If you use gML-sTDA in your research, please cite:

Liu, Xiang-Yang; Wang, Sheng-Rui; Xiao, Dong-Yi; Fang, Wei-Hai; Cui, Ganglong, gML-sTDA: Generative Machine Learning Accelerated sTDA for Excited-State Calculations, https://github.com/I10140317/gML_sTDA

## Contact
For questions or issues, please open a GitHub issue or contact the authors.


