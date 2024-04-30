
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <random>
#include <iomanip> // for std::setprecision

#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "helper_math.h"

#define BLOCK_SIZE 256
#define MAX_FLOAT 3.402823466e+38f      // used for distance comparisons
#define FEMTOSECOND 1e-15f              // 1 femtosecond in seconds
#define ANGSTROM 1e-10f                 // 1 angstrom in meters
#define COULOMB_CONSTANT 8.987551787e9f // Coulomb's constant (N⋅m^2/C^2)
#define GRAVITY 6.67430e-11             // gravitational constant (N⋅m^2⋅kg^−2)
#define BOHR_RADIUS 0.529177f * ANGSTROM

struct Particle
{
    int id;
    float3 position;
    float3 velocity;
    float3 force;
    float mass;
    float charge;
};

__device__ void calculateForces(
    int32_t idx,
    Particle *targets,
    Particle *others,
    int numParticles,
    Particle *sharedParticles)
{
    for (int i{0}; i < numParticles; i += blockDim.x)
    {
        int loadIdx = (i + threadIdx.x);
        if (loadIdx < numParticles)
        {
            sharedParticles[threadIdx.x] = others[loadIdx];
        }
        // sync to ensure all particles have been loaded by all threads
        __syncthreads();

        for (int j{0}; j < blockDim.x && (i + j) < numParticles; ++j)
        {
            Particle &target = targets[idx];
            target.force = make_float3(0.0f, 0.0f, 0.0f);

            // obtain reference to each of the other particles
            Particle &other = sharedParticles[j];

            // calculate vector pointing from the target to the other particle
            float3 distanceVector = other.position - target.position;
            // obtain the magnitude of the distance vector
            float distance = length(distanceVector);
            // divide to obtain the unit vector pointing from the target to the other
            float3 forceDirection = distanceVector / distance;

            // calculate gravitational force
            // F = G * (m1 * m2) / (r^2)
            // float gravMagnitude = GRAVITY * (target.mass * other.mass) / (distance * distance);

            // calculate electrostatic force
            float electroMagnitude = COULOMB_CONSTANT * (fabs(target.charge * other.charge) / (distance * distance));

            // apply the effects of both gravitational and electrostatic forces
            // target.force = forceDirection * (gravMagnitude + electroMagnitude);
            target.force += forceDirection * electroMagnitude;
        }
        // ensure all threads have finished using shared memory before next load
        __syncthreads();
    }
}

__device__ void integrateMotion(
    int idx,
    Particle *particles,
    float deltaTime)
{
    Particle &target = particles[idx];

    float3 acceleration = target.force / target.mass;
    target.velocity += (acceleration * deltaTime);
    target.position += (target.velocity * deltaTime);
}

// ###############################################################################
__global__ void simulationStep(
    Particle *electrons,
    Particle *protons,
    int numParticles,
    float deltaTime)
{
    extern __shared__ Particle sharedParticles[];
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numParticles)
    {
        calculateForces(idx, electrons, protons, numParticles, sharedParticles);
        calculateForces(idx, protons, electrons, numParticles, sharedParticles);
        // synchronize to ensure all forces are updated before modifying motion of particles
        __syncthreads();
        integrateMotion(idx, electrons, deltaTime);
        integrateMotion(idx, protons, deltaTime);
    }
}

__global__ void findNearestProton(
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

    float nearestProtonDistance = MAX_FLOAT;
    int32_t nearestProtonId = -1;

    for (int j{0}; j < numParticles; ++j)
    {
        const Particle &proton = protons[j];

        float3 posDiff = proton.position - electron.position;
        float distance = length(posDiff);

        if (distance < nearestProtonDistance)
        {
            nearestProtonDistance = distance;
            nearestProtonId = proton.id;
        }
    }

    distances[idx] = nearestProtonDistance;
    nearestProtonIds[idx] = nearestProtonId;
}

void hostPrintParticleData(
    const std::vector<Particle> &electrons,
    const std::vector<Particle> &protons,
    int numParticles,
    int step,
    const std::vector<float> &distances,
    const std::vector<int> &nearestProtonIds)
{
    std::cout << "=================== Step " << step << " ===================" << std::endl;
    for (int i = 0; i < numParticles; ++i)
    {
        const Particle &electron = electrons[i];
        const Particle &proton = protons[i];

        float distanceX = proton.position.x - electron.position.x;
        float distanceY = proton.position.y - electron.position.y;
        float distanceZ = proton.position.z - electron.position.z;

        float distance = sqrt(pow(distanceX, 2) + pow(distanceY, 2) + pow(distanceZ, 2));

        std::cout << std::scientific << std::setprecision(3);

        std::cout << "Electron " << electron.id << ":" << std::endl
                  << "\tpos[" << electron.position.x << ", " << electron.position.y << ", " << electron.position.z << "]" << std::endl
                  << "\tvel[" << electron.velocity.x << ", " << electron.velocity.y << ", " << electron.velocity.z << "]" << std::endl
                  << "\tfrc[" << electron.force.x << ", " << electron.force.y << ", " << electron.force.z << "]" << std::endl;

        std::cout
            << "Proton " << proton.id << ":" << std::endl
            << "\tpos[" << proton.position.x << ", " << proton.position.y << ", " << proton.position.z << "]" << std::endl
            << "\tvel[" << proton.velocity.x << ", " << proton.velocity.y << ", " << proton.velocity.z << "]" << std::endl
            << "\tfrc[" << proton.force.x << ", " << proton.force.y << ", " << proton.force.z << "]" << std::endl;

        std::cout << "proton frcDir[" << distanceX / distance << ", " << distanceY / distance << ", " << distanceZ / distance << "]" << std::endl;
        std::cout << "dist = " << distance << std::endl;
    }
    std::cout << "===============================================" << std::endl;
}

int main(int argc, char **argv)
{
    // SIMULATION CONFIGURATION VALUES
    //-------------------------------------------------------------------------------
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
    if (numSteps < 20)
    {
        numSteps = 20;
    }

    deltaTime = std::stof(argv[3]);
    if (deltaTime > 0.001)
    {
        std::cout << "Provided time step is too large and will cause loss of simulation "
                  << "fidelity. Reverting to default of 0.001 femtoseconds." << std::endl;
        deltaTime = 0.001;
    }
    // convert delta time to femtoseconds
    deltaTime *= FEMTOSECOND;

    std::cout << "Configuration received:" << std::endl;
    std::cout << "-----------------------------------------------------------" << std::endl;
    std::cout << "\tNumber of particles in each group = " << numParticlesPerGroup << std::endl;
    std::cout << "\tNumber of steps = " << numSteps << std::endl;
    std::cout << "\tDelta time per step = " << deltaTime << " seconds" << std::endl;
    std::cout << "-----------------------------------------------------------" << std::endl;

    // PARTICLE INSTANTIATION/CONFIGURATION
    //-------------------------------------------------------------------------------
    // create two particle groups: one for electrons and one for protons
    std::vector<Particle> electrons;
    std::vector<Particle> protons;

    // random number generator
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> posRange(0.0f, 0.1f);
    std::uniform_real_distribution<float> velRange(-0.01f, 0.01f);

    electrons.resize(numParticlesPerGroup);
    for (int i{0}; i < numParticlesPerGroup; ++i)
    {
        Particle &e = electrons[i];
        e.id = i;
        e.position = make_float3(posRange(rng) * ANGSTROM,
                                 posRange(rng) * ANGSTROM,
                                 posRange(rng) * ANGSTROM);
        e.velocity = make_float3(velRange(rng) * ANGSTROM / FEMTOSECOND,
                                 velRange(rng) * ANGSTROM / FEMTOSECOND,
                                 velRange(rng) * ANGSTROM / FEMTOSECOND);
        e.force = make_float3(0.0f, 0.0f, 0.0f);

        // e.position = make_float3(BOHR_RADIUS, 0.0f, 0.0f);
        // e.velocity = make_float3(0.0f, 0.0f, 0.0f);
        // e.force = make_float3(0.0f, 0.0f, 0.0f);

        e.mass = 9.10938356e-31f;    // electron mass (kg)
        e.charge = -1.602176634e-19; // Charge of electron (Coulombs)
    }

    protons.resize(numParticlesPerGroup);
    for (int i{0}; i < numParticlesPerGroup; ++i)
    {
        Particle &p = protons[i];
        p.id = i;
        p.position = make_float3(posRange(rng) * ANGSTROM,
                                 posRange(rng) * ANGSTROM,
                                 posRange(rng) * ANGSTROM);
        p.velocity = make_float3(velRange(rng) * ANGSTROM / FEMTOSECOND,
                                 velRange(rng) * ANGSTROM / FEMTOSECOND,
                                 velRange(rng) * ANGSTROM / FEMTOSECOND);
        p.force = make_float3(0.0f, 0.0f, 0.0f);

        // p.position = make_float3(0.0f, 0.0f, 0.0f);
        // p.velocity = make_float3(0.0f, 0.0f, 0.0f);
        // p.force = make_float3(0.0f, 0.0f, 0.0f);

        p.mass = 1.6726219e-27f;    // proton mass (kg)
        p.charge = 1.602176634e-19; // Charge of proton (Coulombs)
    }

    // TEMP: apply an initial velocity to the electron that causes it to orbit the proton
    // float electron_charge = 1.602176634e-19; // electron charge in Coulombs
    // float electron_mass = 9.10938356e-31;    // electron mass in kilograms
    // float r = BOHR_RADIUS;
    // // Velocity for circular orbit at the Bohr radius
    // float v = sqrt((COULOMB_CONSTANT * electron_charge * electron_charge) / (electron_mass * r));
    // // Set the electron's initial velocity to be perpendicular to the radius vector
    // // Assuming the proton is at the origin and the electron is at position (BOHR_RADIUS, 0, 0)
    // electrons[0].velocity = make_float3(0.0f, v, 0.0f);

    // LOG FILE SETUP
    //-------------------------------------------------------------------------------
    // save interval in number of time steps
    // data will be logged to the output file at increments of this value
    int logInterval = 25;

    std::cout << "Creating log file: " << log_name << std::endl;
    std::ofstream file(log_name);
    file << "Step,ElectronID,NearestProtonID,Distance,ElectronPosX,ElectronPosY,ElectronPosZ,NearestProtonPosX,NearestProtonPosY,NearestProtonPosZ\n";

    // vectors that will be used to store distances between particles and the nearest proton for a given electron
    std::vector<float> distances(numParticlesPerGroup);
    std::vector<int> nearestProtonIds(numParticlesPerGroup);

    // DEVICE MEMORY SETUP
    //-------------------------------------------------------------------------------
    size_t particleMem = numParticlesPerGroup * sizeof(Particle);
    size_t floatMem = numParticlesPerGroup * sizeof(float);
    size_t intMem = numParticlesPerGroup * sizeof(int32_t);

    // allocate device memory for particle groups
    Particle *d_electrons;
    Particle *d_protons;
    float *d_distances;
    int32_t *d_nearestProtonIds;

    // malloc for particle array input
    checkCudaErrors(cudaMalloc(&d_electrons, particleMem));
    checkCudaErrors(cudaMalloc(&d_protons, particleMem));

    // copy particle data to device
    checkCudaErrors(cudaMemcpy(d_electrons, electrons.data(), particleMem, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_protons, protons.data(), particleMem, cudaMemcpyHostToDevice));

    // malloc/memset for distances output
    checkCudaErrors(cudaMalloc(&d_distances, floatMem));
    checkCudaErrors(cudaMemset(d_distances, 0, floatMem));
    // malloc/memset for proton IDs output
    checkCudaErrors(cudaMalloc(&d_nearestProtonIds, intMem));
    checkCudaErrors(cudaMemset(d_nearestProtonIds, 0, intMem));

    int blockDim = BLOCK_SIZE;
    int gridDim = (numParticlesPerGroup + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int sharedMemSize = sizeof(Particle) * BLOCK_SIZE;

    // SIMULATION LOOP
    //-------------------------------------------------------------------------------
    std::cout << "Launching simulation..." << std::endl;

    cudaEvent_t cudaStartEvent, cudaStopEvent;
    checkCudaErrors(cudaEventCreate(&cudaStartEvent));
    checkCudaErrors(cudaEventCreate(&cudaStopEvent));

    checkCudaErrors(cudaEventRecord(cudaStartEvent));

    for (uint32_t step{0}; step < numSteps; ++step)
    {
        // launch the findNearestProton kernel at log intervals
        if (step % logInterval == 0)
        {
            findNearestProton<<<gridDim, blockDim>>>(
                d_electrons,
                d_protons,
                numParticlesPerGroup,
                step,
                d_distances,
                d_nearestProtonIds);

            // copy all particle data back to host for logging
            checkCudaErrors(cudaMemcpy(electrons.data(), d_electrons, particleMem, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(protons.data(), d_protons, particleMem, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(distances.data(), d_distances, floatMem, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(nearestProtonIds.data(), d_nearestProtonIds, intMem, cudaMemcpyDeviceToHost));

            // write the resulting data to the log file at each log interval
            for (int i{0}; i < numParticlesPerGroup; ++i)
            {
                const Particle &electron = electrons[i];
                const Particle &proton = protons[nearestProtonIds[i]];
                file << std::scientific << std::setprecision(3)
                     << step << "," << electron.id << "," << proton.id << ","
                     << distances[i] << ","
                     << electron.position.x << "," << electron.position.y << "," << electron.position.z << ","
                     << proton.position.x << "," << proton.position.y << "," << proton.position.z << "\n";
            }
        }

        simulationStep<<<gridDim, blockDim, sharedMemSize>>>(
            d_electrons,
            d_protons,
            numParticlesPerGroup,
            deltaTime);
    }

    checkCudaErrors(cudaEventRecord(cudaStopEvent));
    checkCudaErrors(cudaEventSynchronize(cudaStopEvent));

    float milliseconds = 0;
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, cudaStartEvent, cudaStopEvent));
    std::cout << std::defaultfloat << "Simulation duration = " << milliseconds << "ms" << std::endl;

    // SIMULATION TEARDOWN
    //-------------------------------------------------------------------------------
    file.close();
    // free device memory
    checkCudaErrors(cudaFree(d_electrons));
    checkCudaErrors(cudaFree(d_protons));
    checkCudaErrors(cudaFree(d_distances));
    checkCudaErrors(cudaFree(d_nearestProtonIds));

    std::cout << "Simulation teardown complete." << std::endl;
    return 0;
}