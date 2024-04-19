#include <iostream>
#include <cmath>
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

// float3 operator overloads
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

// Newton's law of universal gravitation, the gravitational force between two particles:
// F = G * (m1 * m2) / (r^2)
// F is the gravitational force between the particles
// G is the gravitational constant
// m1 and m2 are the masses of the two particles
// r is the distance between the particles

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

__global__ void integrateParticles(Particle *particles, int numParticles, float deltaTime)
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

int main()
{
    const int numParticles = 30;
    const float deltaTime = 0.01f;
    const int numSteps = 1000;

    Particle *particles;
    cudaMallocManaged(&particles, numParticles * sizeof(Particle));

    // Initialize particle properties
    for (int i = 0; i < numParticles; ++i)
    {
        particles[i].position = make_float3(rand() / (float)RAND_MAX, rand() / (float)RAND_MAX, rand() / (float)RAND_MAX);
        particles[i].velocity = make_float3(0.0f, 0.0f, 0.0f);
        particles[i].force = make_float3(0.0f, 0.0f, 0.0f);
        particles[i].mass = 1.0f;
        particles[i].charge = (i % 2 == 0) ? 1.0f : -1.0f;
    }

    for (int step = 0; step < numSteps; ++step)
    {
        calculateForces<<<(numParticles + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            particles,
            numParticles,
            deltaTime);

        integrateParticles<<<(numParticles + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            particles,
            numParticles,
            deltaTime);

        cudaDeviceSynchronize();
    }

    // Print final positions of particles
    for (int i = 0; i < numParticles; ++i)
    {
        float forceMagnitude = sqrt(
            particles[i].force.x * particles[i].force.x +
            particles[i].force.y * particles[i].force.y +
            particles[i].force.z * particles[i].force.z);

        std::cout << "Particle_" << i << " pos: "
                  << particles[i].position.x << ", "
                  << particles[i].position.y << ", "
                  << particles[i].position.z << std::endl
                  << "charge: "
                  << particles[i].charge << ", force: "
                  << forceMagnitude << std::endl;
    }

    cudaFree(particles);
    return 0;
}