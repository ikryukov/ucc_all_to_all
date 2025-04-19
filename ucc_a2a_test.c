#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cuda_runtime.h>

#include "ucc_all_to_all.h"


#define CUDA_CHECK(_func)                                           \
    ({                                                              \
        cudaError_t _result = (_func);                              \
        if (cudaSuccess != _result) {                               \
            fprintf(stderr, "%s() failed: %d(%s)",                  \
                    #_func, _result, cudaGetErrorString(_result));  \
            MPI_Abort(MPI_COMM_WORLD, -1);                          \
        }                                                           \
    })

int main(int argc, char **argv) {
    int rank, size;
    int count = 32;
    cudaStream_t stream;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char *v = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    int local_rank = v ? atoi(v) : 0;
    CUDA_CHECK(cudaSetDevice(0)); // TODO: replace to local_rank
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Initialize UCC library
    if (UCC_init() != 0) {
        fprintf(stderr, "Failed to initialize UCC\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int *sbuf, *rbuf;
    int *sbuf_h, *rbuf_h;

    CUDA_CHECK(cudaMalloc((void**)&sbuf, count * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&rbuf, count * sizeof(int)));
    sbuf_h = (int*)malloc(count * sizeof(int));
    rbuf_h = (int*)malloc(count * sizeof(int));

    // Initialize send buffer with rank value
    for (int i = 0; i < count; i++) {
        sbuf_h[i] = rank;
    }

    // Print initial buffer
    printf("Rank %d initial buffer: ", rank);
    for (int j = 0; j < count; j++) {
        printf("%d ", sbuf_h[j]);
    }
    printf("\n");

    memset(rbuf_h, 0, count * sizeof(int));

    CUDA_CHECK(cudaMemcpyAsync(sbuf, sbuf_h, count * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(rbuf, rbuf_h, count * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Perform all-to-all using the library implementation
    if (MPI_Alltoall(sbuf, count, MPI_INT, rbuf, count, MPI_INT, 
                     MPI_COMM_WORLD, stream) != 0) {
        fprintf(stderr, "All-to-all operation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Get results back
    CUDA_CHECK(cudaMemcpyAsync(rbuf_h, rbuf, count * sizeof(int), 
                              cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Print result
    printf("Rank %d result: ", rank);
    for (int j = 0; j < count; j++) {
        printf("%d ", rbuf_h[j]);
    }
    printf("\n");

    // Cleanup
    CUDA_CHECK(cudaFree(sbuf));
    CUDA_CHECK(cudaFree(rbuf));
    free(sbuf_h);
    free(rbuf_h);
    CUDA_CHECK(cudaStreamDestroy(stream));

    // Finalize UCC
    UCC_finalize();

    MPI_Finalize();
    return 0;
}

