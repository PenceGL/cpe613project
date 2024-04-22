# CPE 613 Semester Project, Spring 2024
## N-Body Particle Interaction Simulation
## George Pence

### About The Project
The goal for this project is to create a simple, low-fidelity initial implementation of a
N-body particle simulation with an initial focus on interactions between electrons and
protons for the purpose of analyzing Hydrogen atom formation.

A long-term idea for what this kind of software might evolve into would be a tool that
could, if made efficient and scalable, be used to accelerate the discovery and analysis of
potentially undiscovered materials – or new combinations of materials altogether – more
efficiently than humans can on their own with unassisted methods.

Theoretically, by rapidly simulating various possible combinations and interactions of
particles such as electrons, protons, or even larger, more complex molecules, new
materials with desirable properties such as high strength, thermal stability, or electrical
conductivity could be discovered.

To maintain a realistic scope for this project, the milestone I have set to achieve is to
simulate Hydrogen atom formation through simulating two groups of protons and
electrons and analyzing the resulting data from their simulated interaction.

### Getting Started
This simulation is currently hard-coded to instantiate two groups of particles - electrons and protons - for the purpose simulating hydrogen atom formation as an initial project milestone goal.

### Prerequisites
<!-- Any required libraries or other configuration the project needs to be able to run. -->

### Installation
<!-- Instructions on project setup for the audience -->
No major installation is required.  
Simply change directory into the repository root and then run `make` to build the project executable.

**Note:** The project makefile is configured to use `-arch=sm_89` when invoking NVCC.  
This may need to be updated for your specific GPU architecture.

### Usage
<!-- Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources. -->
Once built, execute a simulation using the following syntax:  
Usage: `./sim_main.exe <num_particles_per_group> <num_steps> <delta_time>`  


### Contact

George Pence - gp0038@uah.edu
