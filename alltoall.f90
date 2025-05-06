program test_mpi_alltoall
    use openacc
    use nccl
    implicit none
    include 'mpif.h'

    integer :: ierr, rank, size, i, j, sendcount
    integer, parameter :: root = 0
    real(8) :: start_time, end_time
    real :: time_taken
    integer(4), allocatable :: sendbuf(:), recvbuf(:)
    integer :: comm
    integer(4) :: iter, niter, sz, nargs
    integer :: NUM_DEVICES, DEVICE_NUM
    CHARACTER(len=32) :: arg
    TYPE(ncclResult)   ncclRes
    TYPE(ncclUniqueId) ncclUID
    TYPE(ncclComm) NCCL_COMM
    real(8) :: mem
    INTERFACE
      SUBROUTINE UCC_INIT() bind(c, name="UCC_init_")
      END SUBROUTINE

      SUBROUTINE UCC_MPI_alltoall(xsend, sendcnt, dataIn, &
                              xrecv, recvcnt, dataOut, &
                              comm, ierr, stream) & 
                 bind(c, name="UCC_Alltoall")
            USE cudafor
            USE iso_c_binding
            integer(c_int), value :: sendcnt, recvcnt
            integer(c_int)        :: ierr
            integer(c_int), value :: dataIn, dataOut, comm
            type(c_devptr), value :: xsend, xrecv
            integer(kind=cuda_stream_kind),value :: stream
      END SUBROUTINE
    END INTERFACE

    ! Initialize MPI
    call MPI_Init(ierr)
    comm = MPI_COMM_WORLD


    ! Get rank and size
    call MPI_Comm_rank(comm, rank, ierr)
    call MPI_Comm_size(comm, size, ierr)

    sz = 1024
    niter = 10
    nargs = command_argument_count()
    if(nargs.ge.1) then
       CALL getarg(1, arg)
       READ(arg, "(I10)") sz
    endif
    if(nargs.ge.2) then
       CALL getarg(2, arg)
       READ(arg, "(I10)") niter 
    endif
!    if(rank == 0) then
!       write(*,*) sz, niter
!    endif
 

    NUM_DEVICES=ACC_GET_NUM_DEVICES(ACC_DEVICE_NVIDIA)
    DEVICE_NUM=MOD(rank,NUM_DEVICES)
    CALL ACC_SET_DEVICE_NUM(DEVICE_NUM,ACC_DEVICE_NVIDIA)


! Init should be called after CUDA initialization
    call UCC_init();

    write (*,*) "Rank ", rank, " uses device ", DEVICE_NUM
    ierr = GET_FREE_MEM_DEVICE(mem)
    write(*,*) DEVICE_NUM, " free mem before NCCL init:", mem

    call MPI_Barrier(comm, ierr)
    start_time= MPI_Wtime()
    IF (rank==0) ncclRes = ncclGetUniqueId(ncclUID)

    CALL MPI_Bcast( ncclUID%internal, 128, MPI_CHAR, 0, MPI_COMM_WORLD, ierr)

    ncclRes = ncclCommInitRank( NCCL_COMM, size, ncclUID, rank)

    IF ( ncclRes == ncclInvalidUsage ) THEN
        write(*,*) "nccl init failed"
    ELSE
        IF ( ncclRes /= ncclSuccess ) &
             write(*,*) 'M_init_nccl: Error in ncclCommInitRank', ncclRes
    ENDIF
!    write(*,*) "nccl_init - done"
    end_time= MPI_Wtime()
    time_taken = end_time - start_time
    if (rank == root) then
         print *, "nccl init: ", time_taken
    end if
    ierr = GET_FREE_MEM_DEVICE(mem)
    write(*,*) DEVICE_NUM, " free mem after NCCL init:", mem



    ! Number of elements each process will send to each other process
    sendcount = sz
    write(*,*) "sendcount = ", sendcount

    ! Allocate send and receive buffers
    allocate(sendbuf(size*sendcount))
    allocate(recvbuf(size*sendcount))

    !$ACC enter data create(sendbuf,recvbuf)
    
    ! Initialize send buffer with some data
    !$ACC PARALLEL LOOP PRESENT(sendbuf,recvbuf) async(1)
    do i = 1, size*sendcount
       sendbuf(i) = rank * 1000 + i
       recvbuf(i) = -1
    end do
    !$ACC wait

    ! Perform MPI_Alltoall and measure the time taken
    call MPI_Barrier(comm, ierr)
    ierr = GET_FREE_MEM_DEVICE(mem)
    write(*,*) DEVICE_NUM, " free mem before warming up:", mem


    !$acc host_data use_device(sendbuf,recvbuf)
    do iter = 0, niter
        ! iter = 0 - warm up
        if(iter .eq. 1) then
            call MPI_Barrier(comm, ierr)
            ierr = GET_FREE_MEM_DEVICE(mem)
            write(*,*) DEVICE_NUM, " free mem before measurments:", mem

            start_time= MPI_Wtime()
        endif
        if( .true. ) then
            ncclRes = ncclGroupStart()
            IF ( ncclRes <> ncclSuccess ) write(*,*) "ncclRes 1:",ncclRes
            DO i=1,size
               ncclRes = ncclSend(c_devloc(sendbuf(1+(i-1)*sendcount)), sendcount, ncclInt, i-1, NCCL_COMM, acc_get_cuda_stream(1))
               IF ( ncclRes <> ncclSuccess ) write(*,*) "ncclRes 2:",ncclRes
               ncclRes = ncclRecv(c_devloc(recvbuf(1+(i-1)*sendcount)), sendcount, ncclInt, i-1, NCCL_COMM, acc_get_cuda_stream(1))
               IF ( ncclRes <> ncclSuccess ) write(*,*) "ncclRes 3:",ncclRes
            ENDDO
            ncclRes = ncclGroupEnd()
            IF ( ncclRes <> ncclSuccess ) write(*,*) "ncclRes 4:",ncclRes
        else
            call UCC_MPI_Alltoall(c_devloc(sendbuf), sendcount, MPI_INTEGER, c_devloc(recvbuf), sendcount, MPI_INTEGER, comm, ierr, acc_get_cuda_stream(1))
            !call MPI_Alltoall(sendbuf, sendcount, MPI_INTEGER, recvbuf, sendcount, MPI_INTEGER, comm, ierr)
        endif
    enddo
    !$acc end host_data
    !$acc wait
    ierr = GET_FREE_MEM_DEVICE(mem)
    write(*,*) DEVICE_NUM, " free mem after measurments:", mem


    end_time= MPI_Wtime()
    time_taken = end_time - start_time

    ! Print the time taken
    if (rank == root) then
        print *, "Time taken for MPI_Alltoall: ", time_taken, " seconds, avg: ", (time_taken/niter) 

    end if

!    !$acc update self(recvbuf, sendbuf)
!        write(*,*) "s", rank, sendbuf
!    write(*,*) "======"
!        write(*,*) "r", rank, recvbuf



    !$ACC exit data delete(sendbuf,recvbuf)

    ncclRes = ncclCommDestroy(NCCL_COMM)

    call MPI_BARRIER(MPI_COMM_WORLD, ierr)
    CALL ACC_SHUTDOWN(ACC_DEVICE_NVIDIA)

    ! Finalize MPI
    call MPI_Finalize(ierr)

    ! Deallocate buffers
    deallocate(sendbuf)
    deallocate(recvbuf)

    contains
  FUNCTION GET_FREE_MEM_DEVICE(MFREE) RESULT(IERR)
    USE openacc, ONLY : ACC_DEVICE_KIND,ACC_GET_DEVICE_TYPE,ACC_DEVICE_NVIDIA,ACC_GET_DEVICE_NUM, &
                        ACC_GET_PROPERTY,ACC_PROPERTY_FREE_MEMORY
    USE iso_c_binding

    REAL(8) :: MFREE
    INTEGER :: IERR

    ! local variables
    INTEGER(KIND=ACC_DEVICE_KIND) :: DEVICE_TYPE
    INTEGER :: IDEVN

    INTEGER(KIND=c_size_t) :: ISIZE

    MFREE=0 ; IERR=0

    DEVICE_TYPE=ACC_GET_DEVICE_TYPE()
    IF (DEVICE_TYPE/=ACC_DEVICE_NVIDIA) THEN
       IERR=1 ; RETURN
    ENDIF

    IDEVN = ACC_GET_DEVICE_NUM(ACC_DEVICE_NVIDIA)
    ISIZE = ACC_GET_PROPERTY(IDEVN,ACC_DEVICE_NVIDIA,ACC_PROPERTY_FREE_MEMORY)

    MFREE = REAL(ISIZE,KIND=8)
  END FUNCTION GET_FREE_MEM_DEVICE



end program test_mpi_alltoall
