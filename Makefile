
all: simulation

simulation: simulation.cu
	nvcc -g -G -arch=sm_89 -I../common simulation.cu -o sim_main

clean:
	rm -f sim_main
