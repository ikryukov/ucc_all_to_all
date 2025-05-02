#!/bin/bash

# CUDA
mpirun --tag-output --mca coll_ucc_enable 0 -x UCX_TLS=^cuda_ipc -x UCC_TLS=cuda -x UCC_COLL_TRACE=debug -x UCC_LOG_LEVEL=debug -x LD_LIBRARY_PATH=/opt/ucc/build-debug/install/lib:$LD_LIBRARY_PATH -np 2 ./ucc_a2a_ftest 2 1

