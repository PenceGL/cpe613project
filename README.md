# CPE 613 Semester Project, Spring 2024
## N-Body Particle Interaction Simulation
## George Pence

### About The Project
The goal for this project was to create a simple, low-fidelity implementation of a
N-body particle simulation with an initial focus on interactions between electrons and
protons for the purpose of attempting to demonstrate and analyze Hydrogen atom formation.

A long-term idea for what this kind of software might evolve into would be a tool that
could, if made efficient and scalable, be used to accelerate the discovery and analysis of
potentially undiscovered materials – or new combinations of existing materials – more
efficiently than humans can on their own with unassisted methods.

Theoretically, by rapidly simulating various possible combinations and interactions of
particles such as electrons, protons, or even larger, more complex molecules, new
materials with desirable properties such as high strength, thermal stability, or electrical
conductivity could be discovered.

To maintain a realistic scope for this project, the milestone set to achieve was to
simulate Hydrogen atom formation through instantiating two groups of protons and
electrons and analyzing the resulting data from their interaction.

### Prerequisites
For this project, I am using the following CUDA version:  
`CUDA compilation tools, release 12.4, V12.4.99 Build cuda_12.4.r12.4/compiler.33961263_0`  
However, the code should compile with any recent CUDA version, so just ensure a version newer than 12.2 is installed.

### Installation
No major installation is required.  
Simply change directory into the repository root and then run `make` to build the project executable.

**Note:** The project Makefile is configured to use `-arch=sm_86` when invoking NVCC.  
This may need to be updated for your specific GPU architecture.

### Usage
Once built, execute a simulation using the following syntax:  
Usage: `./sim_main.exe <num_particles_per_group> <num_steps> <delta_time>`  
Delta time is in femtoseconds. It is recommended to use a delta time value of 0.001 or 0.0001. Anything larger than that will result in the per-step calculations to be too low resolution and a degradation of simulation accuracy.

### Specific Scenarios
To recreate the output of a single stable Hydrogen atom, instantiate a single particle in each group, then modify the initialization code to change the position of the electron to use the following values:  
Position X = 5.29177210903e-11  
Velocity Y = 2187693.75  
All other values should be 0, including the position and velocity of the proton.  

Ensure that the delta time input argument is set to a minimum of 0.001 in order to maintain fidelity. Anything larger will start to lose data resolution and result in incorrect and erratic outputs.  
Using these conditions will set up the initial velocity and position of the electron in such a way that it is more likely to form a stable orbit around the proton.  
After running the simulation, the generated particle_data.csv file can then be scanned in using the `particle_pair_plot.py` Python script in order to generate a static plot over time, or can be fed into the `particle_anim.py` script to generate an actual orbiting animation.  

As can be seen from the resulting plots, the simulation correctly models the electrostatic force between the proton and electron and the electrons orbit around the proton remains stable for any given simulation length.

### Profiling Results
The NSight Compute profiling result files from before and after optimizations were performed to the simulation can be accessed from the following link:  
https://drive.google.com/drive/folders/1pRpXF5DwzKwn2DGSBHDBriNclqns496o?usp=drive_link

### Contact

George Pence - gp0038@uah.edu
