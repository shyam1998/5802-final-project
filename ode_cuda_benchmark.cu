// ode_cuda_benchmark.cu

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <unistd.h> 

#ifndef STATE_DIM
#define STATE_DIM 64
#endif

#ifndef M_TRAJ
#define M_TRAJ 5000000
#endif

#ifndef STEPS
#define STEPS 2000
#endif

#ifndef DT
#define DT 0.0005f
#endif

#ifndef LAMBDA
#define LAMBDA 1.0f
#endif

#define CUDA_CHECK(call) do {                                     \
  cudaError_t err = (call);                                       \
  if (err != cudaSuccess) {                                       \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
            __FILE__, __LINE__, cudaGetErrorString(err));         \
    exit(1);                                                      \
  }                                                               \
} while (0)

__global__ void ode_euler_vec64_inplace(
    float* z,                 // [M * STATE_DIM], updated in-place
    int M,
    int steps,
    float dt,
    float lambda
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M) return;

    float* zi = &z[(size_t)tid * STATE_DIM];

    // dz/dt = -lambda * z  (elementwise)
    for (int s = 0; s < steps; s++) {
        for (int i = 0; i < STATE_DIM; i++) {
            zi[i] += dt * (-lambda * zi[i]);
        }
    }
}

static void log_csv(
    const char* filename,
    int M,
    int steps,
    float dt,
    float lambda,
    double gpu_time_sec,
    double traj_per_sec,
    double checksum,
    double sample_abs_err
) {
    int write_header = (access(filename, F_OK) != 0);

    FILE* f = fopen(filename, "a");
    if (!f) return;

    if (write_header) {
        fprintf(f, "M,steps,dt,lambda,state_dim,gpu_time_sec,traj_per_sec,checksum,sample_abs_err\n");
    }

    fprintf(f, "%d,%d,%.9g,%.9g,%d,%.6f,%.2f,%.6f,%.6e\n",
            M, steps, (double)dt, (double)lambda, STATE_DIM,
            gpu_time_sec, traj_per_sec, checksum, sample_abs_err);

    fclose(f);
}

int main(int argc, char** argv) {
    int M = M_TRAJ;
    int steps = STEPS;
    float dt = DT;
    float lambda = LAMBDA;

 
    if (argc >= 2) M = atoi(argv[1]);
    if (argc >= 3) steps = atoi(argv[2]);
    if (argc >= 4) dt = (float)atof(argv[3]);
    if (argc >= 5) lambda = (float)atof(argv[4]);

    printf("==== CUDA ODE Benchmark (Euler, vec64 in-place) ====\n");
    printf("Trajectories (M) : %d\n", M);
    printf("State dim        : %d\n", STATE_DIM);
    printf("Steps            : %d\n", steps);
    printf("dt               : %.9g\n", (double)dt);
    printf("lambda           : %.9g\n", (double)lambda);

    // Allocate host buffers: z0 and zT (final)
    size_t elems = (size_t)M * STATE_DIM;
    size_t bytes = elems * sizeof(float);

    float* h_z0 = (float*)malloc(bytes);
    float* h_zT = (float*)malloc(bytes);
    if (!h_z0 || !h_zT) {
        fprintf(stderr, "Host malloc failed (bytes=%zu)\n", bytes);
        return 1;
    }

    // Deterministic init: values in (0,1]
    for (int t = 0; t < M; t++) {
        float base = (float)((t % 1000) + 1) / 1000.0f;
        for (int i = 0; i < STATE_DIM; i++) {
            // slightly vary by dimension so checksum isn't trivial
            h_z0[(size_t)t * STATE_DIM + i] = base + 1e-4f * (float)i;
        }
    }

    // Device buffer: in-place state
    float* d_z = nullptr;
    CUDA_CHECK(cudaMalloc(&d_z, bytes));
    CUDA_CHECK(cudaMemcpy(d_z, h_z0, bytes, cudaMemcpyHostToDevice));

    // Warm-up (short run) to stabilize clocks
    {
        int block = 256;
        int grid = (M + block - 1) / block;
        ode_euler_vec64_inplace<<<grid, block>>>(d_z, M, 10, dt, lambda);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // reset state after warmup so benchmark is consistent
        CUDA_CHECK(cudaMemcpy(d_z, h_z0, bytes, cudaMemcpyHostToDevice));
    }

    // Timed run (GPU time via CUDA events)
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int block = 256;
    int grid = (M + block - 1) / block;

    CUDA_CHECK(cudaEventRecord(start));
    ode_euler_vec64_inplace<<<grid, block>>>(d_z, M, steps, dt, lambda);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double sec = (double)ms / 1000.0;

    // Copy back final state
    CUDA_CHECK(cudaMemcpy(h_zT, d_z, bytes, cudaMemcpyDeviceToHost));

    // Checksum (sample ~1000 trajectories, dim 0 only)
    double checksum = 0.0;
    int stride = (M / 1000) + 1;
    for (int t = 0; t < M; t += stride) {
        checksum += (double)h_zT[(size_t)t * STATE_DIM + 0];
    }

    // Analytic check for a single element (trajectory 0, dim 0)
    // True: z(T)=z0*exp(-lambda*T), T=steps*dt
    float z0 = h_z0[0];
    float z_true = z0 * expf(-lambda * (steps * dt));
    float z_num  = h_zT[0];
    double abs_err = (double)fabsf(z_num - z_true);

    double traj_per_sec = (sec > 0.0) ? ((double)M / sec) : 0.0;

    printf("GPU time (s)      : %.6f\n", sec);
    printf("Traj/sec          : %.2f\n", traj_per_sec);
    printf("Checksum (sample) : %.6f\n", checksum);
    printf("Sample abs error  : %.6e\n", abs_err);
    printf("===============================================\n");

    // CSV logging
    log_csv("ode_cuda_results.csv", M, steps, dt, lambda, sec, traj_per_sec, checksum, abs_err);

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_z));
    free(h_z0);
    free(h_zT);

    return 0;
}
