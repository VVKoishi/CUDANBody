// shared memory + loop optimization + multi threads per body

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <iostream>
#include <algorithm>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define SOFTENING 1e-9f

/*
* Each body contains x, y, and z coordinate positions,
* as well as velocities in the x, y, and z directions.
*/
typedef struct { float x, y, z, vx, vy, vz; } Body;

/*
* Do not modify this function. A constraint of this exercise is
* that it remain a host function.
*/

void randomizeBodies(float *data, int n) {
	for (int i = 0; i < n; i++) {
		data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
	}
}

/*
* This function calculates the gravitational impact of all bodies in the system
* on all others, but does not update their positions.
*/

__device__ float3
bodyBodyInteraction(float3 bi, float3 bj, float3 ai)
{
	float3 r;

	// r_ij [3 FLOPS]
	r.x = bj.x - bi.x;
	r.y = bj.y - bi.y;
	r.z = bj.z - bi.z;

	// distSqr = dot(r_ij, r_ij) + EPS^2 [6 FLOPS]
	float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + SOFTENING;

	// invDistCube =1/distSqr^(3/2) [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
	float distSixth = distSqr * distSqr * distSqr;
	float invDistCube = rsqrtf(distSixth);

	// m = 1 
	// s = m_j * invDistCube [1 FLOP]
	// float s = bj.w * invDistCube;

	// a_i = a_i + s * r_ij [6 FLOPS]
	ai.x += r.x * invDistCube;
	ai.y += r.y * invDistCube;
	ai.z += r.z * invDistCube;
	return ai;
}

__device__ float3
tile_calculation(float3 myPosition, float3 accel, int numThreadsPerBody)
{
	int i;
	int p = blockDim.x / numThreadsPerBody;
	int starti = (threadIdx.x / p) * (p / numThreadsPerBody);
	extern __shared__ float3 shPosition[];	// shared memory

	#pragma unroll 32						// loop optimization
	for (i = starti; i < starti + (p / numThreadsPerBody); i++) {
		accel = bodyBodyInteraction(myPosition, shPosition[i], accel);
	}
	return accel;
}

__global__ void
calculate_forces(Body* ptr, float dt, int N, int numThreadsPerBody)
{
	extern __shared__ float3 shPosition[]; // p body shared memory pointer

	int p = blockDim.x / numThreadsPerBody;	// p rows/bodys = num bodys per block
	// int gtid = blockIdx.x * blockDim.x + threadIdx.x; // this thread index
	int gbid = blockIdx.x * p + threadIdx.x % p; // this body index
	float3 myPosition = { ptr[gbid].x, ptr[gbid].y, ptr[gbid].z }; // each body self pos
	float3 acc = { 0.0f, 0.0f, 0.0f }; // each body acc results

	for (int i = 0, int tile = 0; i < N; i += p, tile++) {
		int idx = tile * p + threadIdx.x;
		if (threadIdx.x < p) // divergence but desirable
			shPosition[threadIdx.x] = { ptr[idx].x, ptr[idx].y, ptr[idx].z };
		__syncthreads();
		acc = tile_calculation(myPosition, acc, numThreadsPerBody);
		__syncthreads();
	}
	// Save the result in global memory for the integration step.
	atomicAdd(&ptr[gbid].vx, acc.x * dt);
	atomicAdd(&ptr[gbid].vy, acc.y * dt);
	atomicAdd(&ptr[gbid].vz, acc.z * dt);
}

__global__ void integrate_position(Body *p, float dt, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
    {
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}

int main(const int argc, const char** argv) {

	/*
	 * Do not change the value for `nBodies` here. If you would like to modify it,
	 * pass values into the command line.
	 */

	int nBodies = 4096;
	int salt = 0;
	if (argc > 1) nBodies = 2 << atoi(argv[1]);

	/*
	 * This salt is for assessment reasons. Tampering with it will result in automatic failure.
	 */

	if (argc > 2) salt = atoi(argv[2]);

	const float dt = 0.01f; // time step
	const int nIters = 10;  // simulation iterations

	int bytes = nBodies * sizeof(Body);
	float *buf;

	buf = (float *)malloc(bytes);

	Body *p = (Body*)buf;

	/*
	 * As a constraint of this exercise, `randomizeBodies` must remain a host function.
	 */

	randomizeBodies(buf, 6 * nBodies); // Init pos / vel data

	double totalTime = 0.0;

	auto beginTime = std::chrono::high_resolution_clock::now();

	/*
	 * This simulation will run for 10 cycles of time, calculating gravitational
	 * interaction amongst bodies, and adjusting their positions to reflect.
	 */

	int numBodysPerBlock = 256;
	int numThreadsPerBody = 4;
	
	int blockSize = numBodysPerBlock * numThreadsPerBody;
	int numBlocks = (nBodies + numBodysPerBlock - 1) / numBodysPerBlock;
    int numTiles = numBlocks;
    int sharedMemSize = numBodysPerBlock * 3 * sizeof(float); // 3 floats for pos

	for (int iter = 0; iter < nIters; iter++) {
		calculate_forces <<< numBlocks, blockSize, sharedMemSize >>> (p, dt, nBodies, numThreadsPerBody);
		integrate_position <<< numBlocks, numBodysPerBlock >>> (p, dt, nBodies);
	}

	auto endTime = std::chrono::high_resolution_clock::now();
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - beginTime).count();
	std::cout << ms << "msec\n";

	double gflops = 1e-6 * nBodies * nBodies / ms * 19 * nIters;
	std::cout << gflops << "gflops\n";

	free(buf);
}