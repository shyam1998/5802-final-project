#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <dirent.h>
#include <unistd.h>

#define W 64
#define STRIDE 1
#define MAX_DATASETS 250
#define DATASET_DIR "/home/ssdmw/5802-final-project/datasets/UCR_Anomaly_FullData"

#define TAG_WORK 1
#define TAG_DONE 2
#define TAG_STOP 3


int load_dataset_list(char files[][512]) {
    DIR* dir = opendir(DATASET_DIR);
    if (!dir) return 0;

    struct dirent* e;
    int count = 0;
    while ((e = readdir(dir)) && count < MAX_DATASETS) {
        if (strstr(e->d_name, ".txt")) {
            snprintf(files[count], 512, "%s/%s", DATASET_DIR, e->d_name);
            count++;
        }
    }
    closedir(dir);
    return count;
}

int cmpstr(const void* a, const void* b) {
    const char* sa = (const char*)a;
    const char* sb = (const char*)b;
    return strncmp(sa, sb, 512);
}


float* read_series(const char* path, int* N) {
    FILE* f = fopen(path, "r");
    if (!f) return NULL;

    int cap = 1024, n = 0;
    float* x = malloc(cap * sizeof(float));
    char buf[256];

    while (fgets(buf, sizeof(buf), f)) {
        if (n == cap) {
            cap *= 2;
            x = realloc(x, cap * sizeof(float));
        }
        x[n++] = strtof(buf, NULL);
    }
    fclose(f);
    *N = n;
    return x;
}


void compute_gaf(float* x, float* G) {
    float phi[W];
    for (int i = 0; i < W; i++) {
        float v = fmaxf(-1.f, fminf(1.f, x[i]));
        phi[i] = acosf(v);
    }
    for (int i = 0; i < W; i++)
        for (int j = 0; j < W; j++)
            G[i*W + j] = cosf(phi[i] + phi[j]);
}


void compute_rp(float* x, float* R) {
    for (int i = 0; i < W; i++)
        for (int j = 0; j < W; j++)
            R[i*W + j] = fabsf(x[i] - x[j]);
}

void log_csv(
    const char* filename,
    int ranks,
    int datasets,
    int windows,
    double time_sec,
    double win_per_sec,
    double checksum
) {
    int write_header = access(filename, F_OK) != 0;

    FILE* f = fopen(filename, "a");
    if (!f) return;

    if (write_header) {
        fprintf(f,
            "ranks,datasets,windows,time_sec,windows_per_sec,checksum\n"
        );
    }

    fprintf(f,
        "%d,%d,%d,%.6f,%.2f,%.2f\n",
        ranks,
        datasets,
        windows,
        time_sec,
        win_per_sec,
        checksum
    );

    fclose(f);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0)
            printf("Dynamic scheduling requires np >= 2\n");
        MPI_Finalize();
        return 0;
    }

    char files[MAX_DATASETS][512];
    int num_files = load_dataset_list(files);
    qsort(files, num_files, sizeof(files[0]), cmpstr);

    float G[W*W], R[W*W];
    int local_windows = 0;
    double local_checksum = 0.0;
    double local_time = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    if (rank == 0) {
        int next_file = 0;
        int active_workers = size - 1;

        /* initial dispatch */
        for (int r = 1; r < size && next_file < num_files; r++) {
            MPI_Send(&next_file, 1, MPI_INT, r, TAG_WORK, MPI_COMM_WORLD);
            next_file++;
        }

        /* dynamic scheduling */
        while (active_workers > 0) {
            int worker;
            MPI_Status status;

            MPI_Recv(&worker, 1, MPI_INT, MPI_ANY_SOURCE,
                     TAG_DONE, MPI_COMM_WORLD, &status);

            if (next_file < num_files) {
                MPI_Send(&next_file, 1, MPI_INT, worker,
                         TAG_WORK, MPI_COMM_WORLD);
                next_file++;
            } else {
                MPI_Send(NULL, 0, MPI_INT, worker,
                         TAG_STOP, MPI_COMM_WORLD);
                active_workers--;
            }
        }
    }
    else {
        while (1) {
            int file_id;
            MPI_Status status;

            MPI_Recv(&file_id, 1, MPI_INT, 0,
                     MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TAG_STOP)
                break;

            int N;
            float* x = read_series(files[file_id], &N);
            if (!x) continue;

            int M = (N >= W) ? ((N - W) / STRIDE + 1) : 0;
            for (int w = 0; w < M; w++) {
                int s = w * STRIDE;
                compute_gaf(&x[s], G);
                compute_rp (&x[s], R);
                local_checksum += G[0] + R[0];
                local_windows++;
            }

            free(x);
            MPI_Send(&rank, 1, MPI_INT, 0, TAG_DONE, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    local_time = t1 - t0;

    int total_windows = 0;
    double max_time = 0.0;
    double total_checksum = 0.0;

    MPI_Reduce(&local_windows, &total_windows, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_checksum, &total_checksum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double throughput = total_windows / max_time;
        printf("\n===== GAF/RP MPI Dynamic Scheduling =====\n");
        printf("Sanity Check : %.2f\n", total_checksum);
        printf("Ranks        : %d\n", size);
        printf("Datasets     : %d\n", num_files);
        printf("Windows      : %d\n", total_windows);
        printf("Time (s)     : %.6f\n", max_time);
        printf("Win/sec      : %.2f\n", throughput);
        printf("========================================\n\n");

        log_csv(
        "gaf_rp_dynamic_scaling.csv",
        size,
        num_files,
        total_windows,
        max_time,
        throughput,
        total_checksum
    );
    }

    MPI_Finalize();
    return 0;
}
