# Neural Computation Models

<p align="center">
  <a href="https://gitpoint.co/">
    <img alt="GitPoint" title="GitPoint" src="https://i.postimg.cc/FsqTSYmF/An-LIF-neuron-a-a-schematic-connection-between-three-pre-neurons-to-one-post-neuron.jpg" width="800">
  </a>
</p>

This project implements three neuron models: **LIF (Leaky Integrate-and-Fire)**, **ELIF (Exponential Leaky Integrate-and-Fire)**, and **AELIF (Adaptive Exponential Leaky Integrate-and-Fire)**. These models simulate neuron behavior and analyze their responses to different input currents using the Forward Euler method.

## Table of Contents
- [Project Overview](#project-overview)
- [Implemented Models](#implemented-models)
- [Simulation Inputs](#simulation-inputs)
- [How to Run](#how-to-run)
- [Results](#results)
- [References](#references)

## Project Overview
This project is part of the computational neuroscience course. The goal is to understand and simulate the dynamic behavior of neurons using different computational models. The project includes the implementation of various input current types and analysis of neuron responses.

## Implemented Models
1. **LIF (Leaky Integrate-and-Fire) Model:**  
   This model simulates a basic leaky neuron and captures how its membrane potential evolves over time with different input currents.
   
2. **ELIF (Exponential Leaky Integrate-and-Fire) Model:**  
   An extended version of the LIF model that introduces exponential terms to capture the more complex spiking behavior of neurons.

3. **AELIF (Adaptive Exponential Leaky Integrate-and-Fire) Model:**  
   Adds an adaptation mechanism, modeling how neurons gradually adapt to constant inputs over time.

## Simulation Inputs
The project explores the response of neurons to several types of input currents:
- **Constant Input**
- **Step Input**
- **Sinusoidal Input**
- **Noisy Input**

For each type of input, we simulate the neuron's membrane potential changes and firing rates (frequency-current plots).

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/MohaZamani/Neuron-Models.git
2. Install necessary dependencies:
   ```bash
   pip3 install -r requirements.txt
4. To run the simulations, open the respective Jupyter notebooks for each neuron model:
   - For the AELIF model, open `AELIF_SIM.ipynb`.
   - For the ELIF model, open `ELIF_SIM.ipynb`.
   - For the LIF model, open `LIF_SIM.ipynb`.

   You can run these notebooks using Jupyter by executing:
   ```bash
   jupyter notebook

## Results
Results of the simulations and detailed analysis of the results, refer to the [report](./Report/Report.pdf).

## References
- [PymoNNtorch Framework](https://github.com/cnrl/PymoNNtorch)
- Neural Dynamics online course resources:
   - [Leaky Integrate-and-Fire Neuron](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html)
   - [Exponential Integrate-and-Fire Neuron](https://neuronaldynamics.epfl.ch/online/Ch5.S2.html)
   - [Adaptive Exponential Integrate-and-Fire Neuron](https://neuronaldynamics.epfl.ch/online/Ch6.S1.html)

