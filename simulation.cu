
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <random>

#include <cuda_runtime.h>
#include "helper_cuda.h"

#define BLOCK_SIZE 256
#define MAX_FLOAT 3.402823466e+38f;

struct Particle
{
    int id;
    float3 position;
    float3 velocity;
    float3 force;
    float mass;
    float charge;
};

struct ParticleGroup
{
    std::vector<Particle> particles;
    int numParticles;
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

__device__ float3 operator*(const float &b, const float3 &a)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ float3 operator+(const float3 &a, const float3 &b)
{
    return make_float3((a.x + b.x), (a.y + b.y), (a.z + b.z));
}

__device__ float
getParticleDistance(float3 a, float3 b)
{
    float3 diff = a - b;
    return sqrt((diff.x * diff.x) + (diff.y * diff.y) + (diff.z * diff.z));
}

//-------------------------------------------------------------------------------
__global__ void calculateForces(
    Particle *targetParticles,
    Particle *otherParticles,
    int numParticles,
    int numOtherParticles,
    float deltaTime)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles)
    {
        return;
    }

    // update forces for the target particle group
    Particle &target = targetParticles[idx];
    // the force on each particle is newly calculated on every time step
    target.force = make_float3(0.0f, 0.0f, 0.0f);

    for (int i{0}; i < numParticles; ++i)
    {
        // obtain reference to each of the other particles
        Particle &other = otherParticles[i];
        // calculate position difference between the two particles
        float3 diff = target.position - other.position;

        // calculate distance (magnitude) between particles
        // 1e-8f is added to dist to avoid division by zero
        // in case the particles are extremely close to each other
        float invDist = 1.0f / (getParticleDistance(target.position, other.position) + 1e-8f);

        // obtain the correct direction and magnitude of the acceleration vector
        // by using the cube of the inverse distance
        float invDist3 = invDist * invDist * invDist;

        // calculate and accumulate gravitational force
        float force = target.mass * other.mass * invDist3;
        target.force = target.force + (diff * force);

        // calculate and accumulate electrostatic force (Coulomb's law)
        float k = 8.987e9f;
        float forceElectrostatic = k * target.charge * other.charge * invDist3;
        target.force = target.force + (diff * forceElectrostatic);
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
    int numElectrons,
    int numProtons,
    int step,
    float *distances,
    int *nearestProtonIds)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numElectrons)
        return;

    const Particle &electron = electrons[idx];

    float minDistance = MAX_FLOAT;
    int nearestProtonId = -1;

    for (int j = 0; j < numProtons; ++j)
    {
        const Particle &proton = protons[j];

        float distance = getParticleDistance(electron.position, proton.position);

        if (distance < minDistance)
        {
            minDistance = distance;
            nearestProtonId = proton.id;
        }
    }

    distances[idx] = minDistance;
    nearestProtonIds[idx] = nearestProtonId;
}

int main(int argc, char **argv)
{
    const float FEMTOSECOND = 1e-15f; // 1 femtosecond in seconds
    const float ANGSTROM = 1e-10f;    // 1 angstrom in meters

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
    // enforce a minimum number of particles
    if (numParticlesPerGroup < 20)
    {
        numParticlesPerGroup = 20;
    }

    numSteps = std::stoi(argv[2]);
    // enforce a minimum number of steps
    if (numSteps < 50)
    {
        numSteps = 50;
    }

    deltaTime = std::stof(argv[3]);
    if (deltaTime < 0.1)
    {
        deltaTime = 0.1;
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
    std::vector<ParticleGroup> particleGroups(2);

    // set up random number generator
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> posRange(0.0f, 10.0f);

    // electron group
    particleGroups[0].numParticles = numParticlesPerGroup;
    particleGroups[0].particles.resize(numParticlesPerGroup);
    for (int i = 0; i < numParticlesPerGroup; ++i)
    {
        Particle &p = particleGroups[0].particles[i];
        p.id = i;
        p.position = make_float3(posRange(rng) * ANGSTROM,
                                 posRange(rng) * ANGSTROM,
                                 posRange(rng) * ANGSTROM);
        p.velocity = make_float3(0.0f, 0.0f, 0.0f);
        p.force = make_float3(0.0f, 0.0f, 0.0f);
        p.mass = 9.10938356e-31f; // electron mass (kg)
        p.charge = -1.0f;         // electron charge (atomic units)
    }

    // proton group
    particleGroups[1].numParticles = numParticlesPerGroup;
    particleGroups[1].particles.resize(numParticlesPerGroup);
    for (int i = 0; i < numParticlesPerGroup; ++i)
    {
        Particle &p = particleGroups[1].particles[i];
        p.id = i;
        p.position = make_float3(posRange(rng) * ANGSTROM,
                                 posRange(rng) * ANGSTROM,
                                 posRange(rng) * ANGSTROM);
        p.velocity = make_float3(0.0f, 0.0f, 0.0f);
        p.force = make_float3(0.0f, 0.0f, 0.0f);
        p.mass = 1.6726219e-27f; // proton mass (kg)
        p.charge = 1.0f;         // proton charge (atomic units)
    }

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
    cudaMalloc(&d_electrons, particleGroups[0].numParticles * sizeof(Particle));
    cudaMalloc(&d_protons, particleGroups[1].numParticles * sizeof(Particle));

    // copy particle data from host to device
    cudaMemcpy(d_electrons, particleGroups[0].particles.data(), particleGroups[0].numParticles * sizeof(Particle), cudaMemcpyHostToDevice);
    cudaMemcpy(d_protons, particleGroups[1].particles.data(), particleGroups[1].numParticles * sizeof(Particle), cudaMemcpyHostToDevice);

    // allocate device memory for output arrays
    float *d_distances;
    int *d_nearestProtonIds;
    cudaMalloc(&d_distances, particleGroups[0].numParticles * sizeof(float));
    cudaMalloc(&d_nearestProtonIds, particleGroups[0].numParticles * sizeof(int));

    // SIMULATION LOOP
    //-------------------------------------------------------------------------------
    std::cout << "Launching simulation..." << std::endl;

    int numParticles = particleGroups[0].numParticles;
    int numOtherParticles = particleGroups[1].numParticles;

    int blockDim = BLOCK_SIZE;
    int gridDim = (numParticles + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int step = 0; step < numSteps; ++step)
    {
        calculateForces<<<gridDim, blockDim>>>(
            d_electrons,
            d_protons,
            numParticles,
            numOtherParticles,
            deltaTime);

        calculateForces<<<gridDim, blockDim>>>(
            d_protons,
            d_electrons,
            numParticles,
            numOtherParticles,
            deltaTime);

        integrateParticles<<<gridDim, blockDim>>>(
            d_electrons,
            numParticles,
            deltaTime);

        integrateParticles<<<gridDim, blockDim>>>(
            d_electrons,
            numParticles,
            deltaTime);

        if (step % logInterval == 0)
        {
            // launch the saveParticleData kernel
            saveParticleData<<<(particleGroups[0].numParticles + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                d_electrons,
                d_protons,
                particleGroups[0].numParticles,
                particleGroups[1].numParticles,
                step,
                d_distances,
                d_nearestProtonIds);

            // copy the output arrays from device to host
            std::vector<float> distances(particleGroups[0].numParticles);
            std::vector<int> nearestProtonIds(particleGroups[0].numParticles);
            cudaMemcpy(distances.data(), d_distances, particleGroups[0].numParticles * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(nearestProtonIds.data(), d_nearestProtonIds, particleGroups[0].numParticles * sizeof(int), cudaMemcpyDeviceToHost);

            // write the data to the log file
            for (int i = 0; i < particleGroups[0].numParticles; ++i)
            {
                const Particle &electron = particleGroups[0].particles[i];
                const Particle &proton = particleGroups[1].particles[nearestProtonIds[i]];

                file << step << "," << electron.id << "," << proton.id << ","
                     << distances[i] << ","
                     << electron.position.x << "," << electron.position.y << "," << electron.position.z << ","
                     << proton.position.x << "," << proton.position.y << "," << proton.position.z << "\n";
            }
        }
    }

    // copy updated particle data from device to host
    cudaMemcpy(particleGroups[0].particles.data(), d_electrons, particleGroups[0].numParticles * sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaMemcpy(particleGroups[1].particles.data(), d_protons, particleGroups[1].numParticles * sizeof(Particle), cudaMemcpyDeviceToHost);

    // SIMULATION TEARDOWN
    //-------------------------------------------------------------------------------
    // free device memory
    cudaFree(d_electrons);
    cudaFree(d_protons);
    cudaFree(d_distances);
    cudaFree(d_nearestProtonIds);

    std::cout << "Simulation completed successfully." << std::endl;
    return 0;
}