#!/bin/bash

# CUDA
mpirun --tag-output --mca coll_ucc_enable 0 -x UCX_TLS=^cuda_ipc -x UCC_TLS=cuda -x UCC_COLL_TRACE=debug -x UCC_LOG_LEVEL=debug -x LD_LIBRARY_PATH=/home/ilya/work/ucc/install/lib:$LD_LIBRARY_PATH -np 8 ./ucc_a2a_test

