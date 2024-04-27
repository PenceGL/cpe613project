
all: simulation

simulation: simulation.cu
	nvcc -g -G -arch=sm_89 -I./Includes simulation.cu -o sim_main.exe

clean:
	rm -f sim_main.exe;	rm particle_data.csv
