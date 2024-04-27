
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <random>

#include <cuda_runtime.h>
#include "helper_cuda.h"

#define BLOCK_SIZE 256
#define MAX_FLOAT 3.402823466e+38f
#define FEMTOSECOND 1e-15f // 1 femtosecond in seconds
#define ANGSTROM 1e-10f    // 1 angstrom in meters
#define ANGSTROMSQUARED 1e-20f
#define COLOUMB 8.987551787e9f // Coulomb's constant (N⋅m^2/C^2)

struct Particle
{
    int id;
    float3 position;
    float3 velocity;
    float3 force;
    float mass;
    float charge;
};

// FLOAT3 OPERATOR OVERLOADS
__device__ float3 operator-(const float3 &a, const float3 &b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator*(const float3 &a, const float &b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ float3 operator+(const float3 &a, const float3 &b)
{
    return make_float3((a.x + b.x), (a.y + b.y), (a.z + b.z));
}

// ###############################################################################
__global__ void calculateForces(
    Particle *targetParticles,
    Particle *otherParticles,
    int numParticles,
    float deltaTime)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numParticles)
    {
        return;
    }

    // update forces for the target particle group
    Particle &target = targetParticles[idx];

    for (int i{0}; i < numParticles; ++i)
    {
        // obtain reference to each of the other particles
        Particle &other = otherParticles[i];
        // calculate position difference between the two particles
        float3 distance = other.position - target.position;

        // 1e-8f is added to dist to avoid division by zero
        // in case the particles are extremely close to each other
        float distanceSquared = (distance.x * distance.x) +
                                (distance.y * distance.y) +
                                (distance.z * distance.z) + 1e-8f;

        // calculate distance (magnitude) between particles
        float invDist = 1.0f / sqrtf(distanceSquared);

        // obtain the correct direction and magnitude of the acceleration vector
        // by using the cube of the inverse distance
        float invDist3 = invDist * invDist * invDist;

        // calculate and accumulate gravitational force
        float force = target.mass * other.mass * invDist3;
        target.force = distance * force;

        // calculate and accumulate electrostatic force
        // convert from m^2 to angstrom^2
        float forceElectrostatic = (COLOUMB * target.charge * other.charge) / (distanceSquared * ANGSTROMSQUARED);
        // convert from N to kg⋅angstrom/s^2
        target.force = distance * forceElectrostatic * ANGSTROM;
    }
}

__global__ void integrateParticles(
    Particle *particles,
    int numParticles,
    float deltaTime)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numParticles)
        return;

    Particle &target = particles[idx];

    // update velocity
    // affected by force on the particle over delta time
    target.velocity = target.velocity + (target.force * (deltaTime / target.mass));

    // update position
    // affected by the velocity over the delta time
    target.position = target.position + (target.velocity * deltaTime);
}

__global__ void saveParticleData(
    const Particle *electrons,
    const Particle *protons,
    int numParticles,
    int step,
    float *distances,
    int *nearestProtonIds)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles)
        return;

    const Particle &electron = electrons[idx];

    float minDistance = MAX_FLOAT;
    int nearestProtonId = -1;

    for (int j = 0; j < numParticles; ++j)
    {
        const Particle &proton = protons[j];

        float3 distance = proton.position - electron.position;
        float distanceSquared = (distance.x * distance.x) +
                                (distance.y * distance.y) +
                                (distance.z * distance.z);

        if (distanceSquared < minDistance)
        {
            minDistance = distanceSquared;
            nearestProtonId = proton.id;
        }
    }

    distances[idx] = minDistance;
    nearestProtonIds[idx] = nearestProtonId;
}

int main(int argc, char **argv)
{
    // SIMULATION CONFIGURATION VALUES
    //-------------------------------------------------------------------------------
    int numGroups = 2; // hardcoded number of groups for now
    int numParticlesPerGroup = 0;
    int numSteps = 0;
    float deltaTime = 0.0;
    std::string log_name = "particle_data.csv";

    // ARGUMENT PARSING
    //-------------------------------------------------------------------------------
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <num_particles_per_group> <num_steps> <delta_time>" << std::endl;
        return 1;
    }

    numParticlesPerGroup = std::stoi(argv[1]);
    // enforce a minimum of at least one particle in each group
    if (numParticlesPerGroup < 1)
    {
        numParticlesPerGroup = 1;
    }

    numSteps = std::stoi(argv[2]);
    // enforce a minimum number of steps
    if (numSteps < 50)
    {
        numSteps = 50;
    }

    deltaTime = std::stof(argv[3]);
    if (deltaTime < 0.001)
    {
        deltaTime = 0.001;
    }
    // convert delta time to femtoseconds
    deltaTime *= FEMTOSECOND;

    std::cout << "Configuration received:" << std::endl;
    std::cout << "-----------------------------------------------------------" << std::endl;
    std::cout << "\tNumber of groups = " << numGroups << std::endl;
    std::cout << "\tParticles per group = " << numParticlesPerGroup << std::endl;
    std::cout << "\tNumber of steps = " << numSteps << std::endl;
    std::cout << "\tDelta time per step = " << deltaTime << "femtoseconds" << std::endl;
    std::cout << "-----------------------------------------------------------" << std::endl;

    // PARTICLE CONFIGURATION
    //-------------------------------------------------------------------------------
    // create two particle groups: one for electrons and one for protons
    std::vector<Particle> electrons;
    std::vector<Particle> protons;

    // random number generator
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> posRange(0.0f, 10.0f);

    electrons.resize(numParticlesPerGroup);
    for (int i{0}; i < numParticlesPerGroup; ++i)
    {
        Particle &e = electrons[i];
        e.id = i;
        e.position = make_float3(posRange(rng) * ANGSTROM,
                                 posRange(rng) * ANGSTROM,
                                 posRange(rng) * ANGSTROM);
        e.velocity = make_float3(0.0f, 0.0f, 0.0f);
        e.force = make_float3(0.0f, 0.0f, 0.0f);
        e.mass = 9.10938356e-31f; // electron mass (kg)
        e.charge = -1.0f;         // electron charge (atomic units)

        std::cout << "Electron " << i << " initial position = "
                  << e.position.x << ", "
                  << e.position.y << ", "
                  << e.position.z << std::endl;
    }

    protons.resize(numParticlesPerGroup);
    for (int i{0}; i < numParticlesPerGroup; ++i)
    {
        Particle &p = protons[i];
        p.id = i;
        p.position = make_float3(posRange(rng) * ANGSTROM,
                                 posRange(rng) * ANGSTROM,
                                 posRange(rng) * ANGSTROM);
        p.velocity = make_float3(0.0f, 0.0f, 0.0f);
        p.force = make_float3(0.0f, 0.0f, 0.0f);
        p.mass = 1.6726219e-27f; // proton mass (kg)
        p.charge = 1.0f;         // proton charge (atomic units)

        std::cout << "Proton " << i << " initial position = "
                  << p.position.x << ", "
                  << p.position.y << ", "
                  << p.position.z << std::endl;
    }

    float distanceX = protons[0].position.x - electrons[0].position.x;
    float distanceY = protons[0].position.y - electrons[0].position.y;
    float distanceZ = protons[0].position.z - electrons[0].position.z;

    float distanceSquared = (distanceX * distanceX) +
                            (distanceY * distanceY) +
                            (distanceZ * distanceZ);
    std::cout << "Initial distance between particles = "
              << distanceX << ", "
              << distanceY << ", "
              << distanceZ << std::endl;
    std::cout << "Initial distance squared between particles = " << distanceSquared << std::endl;

    // LOG FILE SETUP
    //-------------------------------------------------------------------------------
    // save interval in number of time steps
    // data will be logged to the output file at increments of this value
    int logInterval = 100;

    std::cout << "Creating log file: " << log_name << std::endl;
    std::ofstream file(log_name);
    file << "Step,ElectronID,NearestProtonID,Distance,ElectronPosX,ElectronPosY,ElectronPosZ,NearestProtonPosX,NearestProtonPosY,NearestProtonPosZ\n";

    // DEVICE MEMORY SETUP
    //-------------------------------------------------------------------------------
    // allocate device memory for particle groups
    Particle *d_electrons;
    Particle *d_protons;
    checkCudaErrors(cudaMalloc(&d_electrons, numParticlesPerGroup * sizeof(Particle)));
    checkCudaErrors(cudaMalloc(&d_protons, numParticlesPerGroup * sizeof(Particle)));

    // copy particle data from host to device
    checkCudaErrors(cudaMemcpy(d_electrons, electrons.data(), numParticlesPerGroup * sizeof(Particle), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_protons, protons.data(), numParticlesPerGroup * sizeof(Particle), cudaMemcpyHostToDevice));

    // allocate device memory for output arrays
    float *d_distances;
    int *d_nearestProtonIds;
    checkCudaErrors(cudaMalloc(&d_distances, numParticlesPerGroup * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_nearestProtonIds, numParticlesPerGroup * sizeof(int)));

    // SIMULATION LOOP
    //-------------------------------------------------------------------------------
    std::cout << "Launching simulation..." << std::endl;

    int blockDim = BLOCK_SIZE;
    int gridDim = (numParticlesPerGroup + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int step = 0; step < numSteps; ++step)
    {
        calculateForces<<<gridDim, blockDim>>>(
            d_electrons,
            d_protons,
            numParticlesPerGroup,
            deltaTime);

        calculateForces<<<gridDim, blockDim>>>(
            d_protons,
            d_electrons,
            numParticlesPerGroup,
            deltaTime);

        // calculate all forces prior to integrating
        checkCudaErrors(cudaDeviceSynchronize());

        integrateParticles<<<gridDim, blockDim>>>(
            d_electrons,
            numParticlesPerGroup,
            deltaTime);

        integrateParticles<<<gridDim, blockDim>>>(
            d_protons,
            numParticlesPerGroup,
            deltaTime);

        // integrate all particles prior to logging
        checkCudaErrors(cudaDeviceSynchronize());

        // launch the saveParticleData kernel at log intervals
        if (step % logInterval == 0)
        {
            saveParticleData<<<gridDim, blockDim>>>(
                d_electrons,
                d_protons,
                numParticlesPerGroup,
                step,
                d_distances,
                d_nearestProtonIds);

            // copy the output arrays from device to host
            std::vector<float> distances(numParticlesPerGroup);
            std::vector<int> nearestProtonIds(numParticlesPerGroup);

            checkCudaErrors(cudaMemcpy(distances.data(), d_distances, numParticlesPerGroup * sizeof(float), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(nearestProtonIds.data(), d_nearestProtonIds, numParticlesPerGroup * sizeof(int), cudaMemcpyDeviceToHost));

            // write the data to the log file
            for (int i{0}; i < numParticlesPerGroup; ++i)
            {
                const Particle &electron = electrons[i];
                const Particle &proton = protons[nearestProtonIds[i]];

                file << step << "," << electron.id << "," << proton.id << ","
                     << distances[i] << ","
                     << electron.position.x << "," << electron.position.y << "," << electron.position.z << ","
                     << proton.position.x << "," << proton.position.y << "," << proton.position.z << "\n";
            }
        }
    }

    // SIMULATION TEARDOWN
    //-------------------------------------------------------------------------------
    // free device memory
    checkCudaErrors(cudaFree(d_electrons));
    checkCudaErrors(cudaFree(d_protons));
    checkCudaErrors(cudaFree(d_distances));
    checkCudaErrors(cudaFree(d_nearestProtonIds));

    std::cout << "Simulation completed successfully." << std::endl;
    return 0;
}