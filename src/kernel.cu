#include <iostream>
#include "kernel.h"
#include <device_launch_parameters.h>

struct SimBox {
    double3 box; 
    double3 boxh;
    double   a;
    int      dim_n;
    int      n2; 
};

// device-side accumulator for centre-of-mass velocity
__device__ double d_cm[3]; 

__global__ void init_pos_vel(double3* x, double3* v,
                             SimBox sb, double sigma,
                             unsigned long long seed, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;

    /* ------------ velocities (Gaussian) ------------- */
    curandStatePhilox4_32_10_t st;
    curand_init(seed, i, 0, &st);

    double3 vi;
    vi.x = sigma * curand_normal_double(&st);
    vi.y = sigma * curand_normal_double(&st);
    vi.z = sigma * curand_normal_double(&st);
    v[i] = vi;

    // atomic CM accumulator
    atomicAdd(&d_cm[0], vi.x);
    atomicAdd(&d_cm[1], vi.y);
    atomicAdd(&d_cm[2], vi.z);

    /* ------------ positions (simple cubic) ----------- */
    int n  = sb.dim_n;
    int n2 = sb.n2;
    int ix =  i % n;
    int iy = (i % n2) / n;
    int iz =  i / n2;

    double3 xi;
    xi.x =  ix * sb.a + 0.5*sb.a - sb.boxh.x;
    xi.y =  iy * sb.a + 0.5*sb.a - sb.boxh.y;
    xi.z =  iz * sb.a + 0.5*sb.a - sb.boxh.z;
    x[i] = xi;
}

__global__ void remove_cm(double3* v, int N)
{
    // compute CM once per block read-only
    __shared__ double3 cm;
    if (threadIdx.x == 0) {
        cm.x = d_cm[0] / N;
        cm.y = d_cm[1] / N;
        cm.z = d_cm[2] / N;
    }
    __syncthreads();

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;
    v[i].x -= cm.x;
    v[i].y -= cm.y;
    v[i].z -= cm.z;
}
// -----------------------------------------------------------------------

void initCUDA(int N){
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
}

void stepSimulation(double dt){

}