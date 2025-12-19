// ode_cpu_baseline.c

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#define STATE_DIM 64

double wall_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void log_csv(
    const char* filename,
    int M,
    int steps,
    float dt,
    float lambda,
    double runtime,
    double traj_per_sec,
    double checksum
) {
    int write_header = access(filename, F_OK) != 0;

    FILE* f = fopen(filename, "a");
    if (!f) return;

    if (write_header) {
        fprintf(f,
            "M,steps,dt,lambda,state_dim,cpu_time_sec,traj_per_sec,checksum\n"
        );
    }

    fprintf(f,
        "%d,%d,%.9g,%.9g,%d,%.6f,%.2f,%.6f\n",
        M, steps, (double)dt, (double)lambda, STATE_DIM,
        runtime, traj_per_sec, checksum
    );

    fclose(f);
}

int main(int argc, char** argv) {
    int M = 500000;
    int steps = 2000;
    float dt = 0.0005f;
    float lambda = 1.0f;

    if (argc >= 2) M = atoi(argv[1]);
    if (argc >= 3) steps = atoi(argv[2]);
    if (argc >= 4) dt = atof(argv[3]);
    if (argc >= 5) lambda = atof(argv[4]);

    printf("==== CPU ODE Baseline (Euler) ====\n");
    printf("Trajectories (M): %d\n", M);
    printf("State dim       : %d\n", STATE_DIM);
    printf("Steps           : %d\n", steps);

    size_t elems = (size_t)M * STATE_DIM;
    float* z = (float*)malloc(elems * sizeof(float));
    if (!z) {
        printf("malloc failed\n");
        return 1;
    }

    // Init
    for (int t = 0; t < M; t++) {
        float base = (float)((t % 1000) + 1) / 1000.0f;
        for (int i = 0; i < STATE_DIM; i++) {
            z[t * STATE_DIM + i] = base + 1e-4f * i;
        }
    }

    double t0 = wall_time();

    // Euler integration
    for (int s = 0; s < steps; s++) {
        for (int t = 0; t < M; t++) {
            float* zi = &z[t * STATE_DIM];
            for (int i = 0; i < STATE_DIM; i++) {
                zi[i] += dt * (-lambda * zi[i]);
            }
        }
    }

    double t1 = wall_time();
    double runtime = t1 - t0;

    // Checksum
    double checksum = 0.0;
    int stride = (M / 1000) + 1;
    for (int t = 0; t < M; t += stride) {
        checksum += z[t * STATE_DIM];
    }

    double traj_per_sec = M / runtime;

    printf("CPU time (s)     : %.6f\n", runtime);
    printf("Traj/sec         : %.2f\n", traj_per_sec);
    printf("Checksum         : %.6f\n", checksum);
    printf("=================================\n");

    // CSV logging
    log_csv(
        "ode_cpu_results.csv",
        M,
        steps,
        dt,
        lambda,
        runtime,
        traj_per_sec,
        checksum
    );

    free(z);
    return 0;
}
