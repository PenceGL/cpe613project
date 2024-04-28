
all: simulation

simulation: simulation.cu
	nvcc -g -arch=sm_86 -lineinfo -I./Includes simulation.cu -o sim_main.exe

clean:
	rm -f sim_main.exe;	rm -f particle_data.csv

# nsys profile -o simprofile ./sim_main.exe 0 1000 0.001
# ncu -f -o simcompute --set full --import-source yes ./sim_main.exe 0 1000 0.001