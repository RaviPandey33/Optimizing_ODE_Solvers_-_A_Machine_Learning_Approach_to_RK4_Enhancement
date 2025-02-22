# Optimizing ODE Solvers: A Machine Learning Approach to RK4 Enhancement

Welcome to the GitHub repository for my thesis project on enhancing Runge-Kutta 4 (RK4) methods using machine learning techniques. This repository contains all the code and results files used and generated during my research.

## Repository Structure

# Implementation Directory

This directory contains the core computational notebooks and scripts used for the thesis project on optimizing Runge-Kutta methods for Hamiltonian systems. Two key notebooks are central to understanding the optimization results and methodology:

## 1. Final_Plots.ipynb

### Description
This notebook presents dynamic and energy plots for the Hamiltonian systems analyzed in the thesis. It visualizes the system's behavior both before and after the optimization process, showcasing the effectiveness of the applied optimizations.

### Contents
- **Dynamics Plots**: These plots illustrate the trajectory of the system over time, demonstrating how the system evolves.
- **Energy Plots**: These plots show the energy conservation in the system before and after the optimization process, highlighting the improvements in energy efficiency due to optimization.

## 2. Error_Convergence_Plot.ipynb

### Description
This notebook focuses on the error convergence associated with the optimization process of the Runge-Kutta methods. It provides a detailed analysis of how the errors reduce as the optimization progresses.

### Contents
- **Convergence Plots**: Detailed plots showing the error reduction through successive iterations of the optimization process.
- **Visual Comparisons**: These include comparative visuals of error rates before and after the optimization, offering a clear depiction of the optimization's impact.



### Halton Sequence

The `Halton Sequence` folder focuses on the implementation of the Halton sequence and its comparison to JAX random generation methods. The goal here is to illustrate the distribution and quality of points generated in a 2D [0,1] space, which is crucial for understanding the initial conditions used in the simulations.

#### Key Features:
- **Halton vs. JAX Random Points**: Compares the uniformity and distribution of points generated by the Halton sequence against those generated using JAX's random module.
- **Visualizations**: Includes plots that clearly show the spatial distribution of points, providing insight into the sequence's effectiveness for numerical simulations.




