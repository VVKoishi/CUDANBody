// shared memory + loop optimization

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
tile_calculation(float3 myPosition, float3 accel, float3 localShufflePos)
{
	// every 32 threads warp do 32 times of shuffle
	// to use the register "localShufflePos"
	#pragma unroll 32
	for (int j = 0; j < 32; j++) {
		// get the shuffle data
		float shuffle_x = __shfl_sync(0xFFFFFFFF, localShufflePos.x, j);
		float shuffle_y = __shfl_sync(0xFFFFFFFF, localShufflePos.y, j);
		float shuffle_z = __shfl_sync(0xFFFFFFFF, localShufflePos.z, j);
		// use the shuffle data
		accel = bodyBodyInteraction(myPosition, localShufflePos, accel);
	}

	return accel;
}

__global__ void
calculate_forces(Body* ptr, float dt, int N)
{

	int p = blockDim.x;	// 32
	int gtid = blockIdx.x * blockDim.x + threadIdx.x; // this thread/body index
	Body myBody = ptr[gtid];
	float3 myPosition = { myBody.x, myBody.y, myBody.z }; // each body self pos
	float3 acc = { 0.0f, 0.0f, 0.0f }; // each body acc results

	for (int i = 0, int tile = 0; i < N; i += p, tile++) {
		int idx = tile * blockDim.x + threadIdx.x;
		Body localBody = ptr[idx];
		//__syncthreads();
		acc = tile_calculation(myPosition, acc, { localBody.x, localBody.y, localBody.z });
		//__syncthreads();
	}
	// Save the result in global memory for the integration step.
	ptr[gtid].vx += acc.x * dt;
	ptr[gtid].vy += acc.y * dt;
	ptr[gtid].vz += acc.z * dt;
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

	int blockSize = 32; // only can be 32
	int numBlocks = (nBodies + blockSize - 1) / blockSize;
    int numTiles = numBlocks;

	for (int iter = 0; iter < nIters; iter++) {
		calculate_forces <<< numBlocks, blockSize >>> (p, dt, nBodies);
		integrate_position <<< numBlocks, blockSize >>> (p, dt, nBodies);
	}

	auto endTime = std::chrono::high_resolution_clock::now();
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - beginTime).count();
	std::cout << ms << "msec\n";

	double gflops = 1e-6 * nBodies * nBodies / ms * 19 * nIters;
	std::cout << gflops << "gflops\n";

	free(buf);
}