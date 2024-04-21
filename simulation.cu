
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

#include <cuda_runtime.h>

#define BLOCK_SIZE 256

struct Particle
{
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

//-------------------------------------------------------------------------------
__global__ void calculateForces(
    Particle *particles,
    int numParticles,
    float deltaTime)
{
    // verify that the thread being used does not exceed the number of particles
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles)
    {
        return;
    }

    // obtain reference to current particle
    Particle &p = particles[idx];
    p.force = make_float3(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < numParticles; ++i)
    {
        if (i == idx)
        {
            // skip making the particle interact with itself
            continue;
        }

        // obtain reference to other particles
        Particle &q = particles[i];
        // calculate position difference between the two particles
        float3 diff = q.position - p.position;
        // calculate distance (magnitude) between particles
        float dist = sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
        // obtain the inverse of the distance
        // 1e-5f is added to dist to avoid division by zero
        // in case the particles are extremely close to each other
        float invDist = 1.0f / (dist + 1e-5f);

        // obtain the correct direction and magnitude of the acceleration vector
        // by using the cube of the inverse distance
        float invDist3 = invDist * invDist * invDist;

        // gravitational force calculation
        float force = p.mass * q.mass * invDist3;
        p.force = p.force + (diff * force);

        // electrostatic force (Coulomb's law)
        float k = 8.99e9f; // Coulomb's constant (N⋅m²/C²)
        float forceElectrostatic = k * p.charge * q.charge * invDist3;
        p.force = p.force + (diff * forceElectrostatic);
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

    Particle &p = particles[idx];

    // Update velocity
    p.velocity = p.velocity + (p.force * (deltaTime / p.mass));

    // Update position
    p.position = p.position + (p.velocity * deltaTime);
}

void saveParticleData(
    const std::vector<ParticleGroup> &particleGroups,
    int step,
    std::ofstream &file)
{
    for (int g = 0; g < particleGroups.size(); ++g)
    {
        const ParticleGroup &group = particleGroups[g];
        for (int i = 0; i < group.numParticles; ++i)
        {
            const Particle &p = group.particles[i];
            float forceMagnitude = sqrt(p.force.x * p.force.x + p.force.y * p.force.y + p.force.z * p.force.z);

            file << step << "," << g << "," << i << ","
                 << p.position.x << "," << p.position.y << "," << p.position.z << ","
                 << p.velocity.x << "," << p.velocity.y << "," << p.velocity.z << ","
                 << p.charge << "," << forceMagnitude << "\n";
        }
    }
}

int main(int argc, char **argv)
{
    const float FEMTOSECOND = 1e-15f; // 1 femtosecond in seconds
    const float ANGSTROM = 1e-10f;    // 1 angstrom in meters

    // SIMULATION CONFIGURATION VALUES
    //-------------------------------------------------------------------------------
    int numGroups = 2;
    int numParticlesPerGroup;
    std::string log_name = "particle_data.csv";

    // ARG PARSING
    //-------------------------------------------------------------------------------
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <num_particles_per_group> <num_steps> <delta_time>" << std::endl;
        return 1;
    }

    // int numGroups = std::stoi(argv[1]);

    if (!std::stoi(argv[1]) >= 100)
    {
        numParticlesPerGroup = 100;
    }
    int numSteps = std::stoi(argv[2]);
    float deltaTime = std::stof(argv[3]) * FEMTOSECOND;

    std::cout << "Configuration received:" << std::endl;
    std::cout << "Number of groups = " << numGroups
              << ", particles per group = " << numParticlesPerGroup << std::endl;
    std::cout << "Number of steps = " << numSteps
              << ", delta time = " << deltaTime << "femtoseconds" << std::endl;

    // PARTICLE CONFIGURATION
    //-------------------------------------------------------------------------------
    std::vector<ParticleGroup> particleGroups(2); // Create two particle groups: electrons and protons

    // electron group
    particleGroups[0].numParticles = numParticlesPerGroup;
    particleGroups[0].particles.resize(numParticlesPerGroup);
    for (int i = 0; i < numParticlesPerGroup; ++i)
    {
        Particle &p = particleGroups[0].particles[i];
        p.position = make_float3(rand() / (float)RAND_MAX * 10.0f * ANGSTROM,
                                 rand() / (float)RAND_MAX * 10.0f * ANGSTROM,
                                 rand() / (float)RAND_MAX * 10.0f * ANGSTROM);
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
        p.position = make_float3(rand() / (float)RAND_MAX * 10.0f * ANGSTROM,
                                 rand() / (float)RAND_MAX * 10.0f * ANGSTROM,
                                 rand() / (float)RAND_MAX * 10.0f * ANGSTROM);
        p.velocity = make_float3(0.0f, 0.0f, 0.0f);
        p.force = make_float3(0.0f, 0.0f, 0.0f);
        p.mass = 1.6726219e-27f; // proton mass (kg)
        p.charge = 1.0f;         // proton charge (atomic units)
    }

    // SIMULATION CONFIGURATION
    //-------------------------------------------------------------------------------
    int saveInterval = 100; // save data every 100 steps

    std::cout << "Creating log file: " << log_name << std::endl;
    std::ofstream file(log_name);
    file << "Step,Group,ParticleID,PositionX,PositionY,PositionZ,VelocityX,VelocityY,VelocityZ,Charge,ForceMagnitude\n";

    // SIMULATION LOOP
    //-------------------------------------------------------------------------------
    for (int step = 0; step < numSteps; ++step)
    {
        for (int g = 0; g < numGroups; ++g)
        {
            ParticleGroup &group = particleGroups[g];
            Particle *d_particles;
            cudaMalloc(&d_particles, group.numParticles * sizeof(Particle));
            cudaMemcpy(d_particles, group.particles.data(), group.numParticles * sizeof(Particle), cudaMemcpyHostToDevice);

            calculateForces<<<(group.numParticles + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                d_particles,
                group.numParticles,
                deltaTime);

            integrateParticles<<<(group.numParticles + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                d_particles,
                group.numParticles,
                deltaTime);

            cudaMemcpy(group.particles.data(), d_particles, group.numParticles * sizeof(Particle), cudaMemcpyDeviceToHost);
            cudaFree(d_particles);
        }

        if (step % saveInterval == 0)
        {
            saveParticleData(particleGroups, step, file);
        }
    }

    // SIMULATION END
    //-------------------------------------------------------------------------------

    return 0;
}