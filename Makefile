# Compiler and flags
MPICC = mpicxx
MPIF90 = mpif90
NVCC = nvcc
CUDA_HOME = /opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda/12.8
# CUDA_HOME = /usr/local/cuda
UCC_HOME = /home/ilya/work/ucc/install

# Include paths
INCLUDES = -I$(CUDA_HOME)/include \
           -I$(UCC_HOME)/include \
           -I./include

# Compiler flags
CFLAGS =  -fPIC $(INCLUDES)
FFLAGS =  -acc -gpu=cc80,cc90,cc70 -Minfo=accel
NVCCFLAGS = -g -O2 $(INCLUDES)

# Linker flags
LDFLAGS = -L$(CUDA_HOME)/lib64 -L$(UCC_HOME)/lib -acc 
LIBS = -lcudart -lucc -lnvToolsExt

# Library name
LIB = libucc_a2a.so

# Target executable
TARGET = ucc_a2a_test 
TARGETF = ucc_a2a_ftest

# Source files
LIB_SRC = src/ucc_all_to_all.c
TEST_SRC = ucc_a2a_test.c
TEST_FSRC = alltoall.f90

# Object files
LIB_OBJ = $(LIB_SRC:.c=.o)
TEST_OBJ = $(TEST_SRC:.c=.o)

# Default target
all: $(LIB) $(TARGET) $(TARGETF)

# Library target
$(LIB): $(LIB_OBJ)
	$(MPICC) -shared $(LIB_OBJ) $(LDFLAGS) $(LIBS) -o $@

# Test executable target
$(TARGET): $(TEST_OBJ) $(LIB)
	$(MPICC) $(TEST_OBJ) -L. -lucc_a2a $(LDFLAGS) $(LIBS) -o $@


$(TARGETF): alltoall.o  $(LIB)
	$(MPIF90) $< -L. -lucc_a2a $(LDFLAGS) $(LIBS) -cudalib=nccl -o $@	

# Compilation rules
%.o: %.c
	$(MPICC) $(CFLAGS) -c $< -o $@

%.o: %.f90
	$(MPIF90) $(FFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(TARGET) $(TARGETF) $(LIB) *.o *~ 

.PHONY: all clean
