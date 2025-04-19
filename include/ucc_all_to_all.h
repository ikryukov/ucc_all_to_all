#pragma once

#include <mpi.h>
#include <cuda_runtime.h>

int UCC_init();
int UCC_finalize();

int MPI_Alltoall(const void *sendbuf, int sendcount,
     MPI_Datatype sendtype, void *recvbuf, int recvcount,
     MPI_Datatype recvtype, MPI_Comm comm, cudaStream_t stream);
