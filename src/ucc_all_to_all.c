#include "ucc_all_to_all.h"

#include <ucc/api/ucc.h>

#define STR(x) #x
#define UCC_CHECK(_call)                                            \
    if (UCC_OK != (_call)) {                                        \
        fprintf(stderr, "*** UCC TEST FAIL: %s\n", STR(_call));     \
        MPI_Abort(MPI_COMM_WORLD, -1);                              \
    }

static ucc_datatype_t mpi_to_ucc_datatype(MPI_Datatype dt) {
    if (dt == MPI_CHAR)               return UCC_DT_INT8;
    if (dt == MPI_SIGNED_CHAR)        return UCC_DT_INT8;
    if (dt == MPI_UNSIGNED_CHAR)      return UCC_DT_UINT8;
    if (dt == MPI_BYTE)               return UCC_DT_INT8;
    if (dt == MPI_SHORT)              return UCC_DT_INT16;
    if (dt == MPI_UNSIGNED_SHORT)     return UCC_DT_UINT16;
    if (dt == MPI_INT)                return UCC_DT_INT32;
    if (dt == MPI_UNSIGNED)           return UCC_DT_UINT32;
    if (dt == MPI_LONG)               return UCC_DT_INT64;
    if (dt == MPI_UNSIGNED_LONG)      return UCC_DT_UINT64;
    if (dt == MPI_FLOAT)              return UCC_DT_FLOAT32;
    if (dt == MPI_DOUBLE)             return UCC_DT_FLOAT64;
    if (dt == MPI_LONG_DOUBLE)        return UCC_DT_FLOAT128;
    if (dt == MPI_COMPLEX)            return UCC_DT_FLOAT32_COMPLEX;
    if (dt == MPI_DOUBLE_COMPLEX)     return UCC_DT_FLOAT64_COMPLEX;
    if (dt == MPI_LONG_LONG)          return UCC_DT_INT64;
    if (dt == MPI_UNSIGNED_LONG_LONG) return UCC_DT_UINT64;
    
    fprintf(stderr, "Unsupported MPI datatype in conversion to UCC\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 0; // Never reached
}

static ucc_status_t oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                                  void *coll_info, void **req)
{
    MPI_Comm    comm = (MPI_Comm)coll_info;
    MPI_Request request;

    MPI_Iallgather(sbuf, msglen, MPI_BYTE, rbuf, msglen, MPI_BYTE, comm,
                   &request);
    *req = (void *)request;
    return UCC_OK;
}

static ucc_status_t oob_allgather_test(void *req)
{
    MPI_Request request = (MPI_Request)req;
    int         completed;

    MPI_Test(&request, &completed, MPI_STATUS_IGNORE);
    return completed ? UCC_OK : UCC_INPROGRESS;
}

static ucc_status_t oob_allgather_free(void *req)
{
    return UCC_OK;
}

/* Creates UCC team for a group of processes represented by MPI
   communicator. UCC API provides different ways to create a team,
   one of them is to use out-of-band (OOB) allgather provided by
   the calling runtime. */
static ucc_team_h create_ucc_team(MPI_Comm comm, ucc_context_h ctx)
{
    int               rank, size;
    ucc_team_h        team;
    ucc_team_params_t team_params;
    ucc_status_t      status;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    team_params.mask          = UCC_TEAM_PARAM_FIELD_OOB;
    team_params.oob.allgather = oob_allgather;
    team_params.oob.req_test  = oob_allgather_test;
    team_params.oob.req_free  = oob_allgather_free;
    team_params.oob.coll_info = (void*)comm;
    team_params.oob.n_oob_eps = size;
    team_params.oob.oob_ep    = rank;

    UCC_CHECK(ucc_team_create_post(&ctx, 1, &team_params, &team));
    while (UCC_INPROGRESS == (status = ucc_team_create_test(team))) {
        UCC_CHECK(ucc_context_progress(ctx));
    };
    if (UCC_OK != status) {
        fprintf(stderr, "failed to create ucc team\n");
        MPI_Abort(MPI_COMM_WORLD, status);
    }
    return team;
}

ucc_context_h g_ctx;
ucc_lib_h     g_lib;

ucc_team_h g_team;
int g_rank, g_size;

int UCC_init() {

    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);

    /* Init ucc library */
    ucc_lib_params_t lib_params = {
        .mask        = UCC_LIB_PARAM_FIELD_THREAD_MODE,
        .thread_mode = UCC_THREAD_SINGLE
    };
    ucc_lib_config_h     lib_config;

    UCC_CHECK(ucc_lib_config_read(NULL, NULL, &lib_config));
    UCC_CHECK(ucc_init(&lib_params, lib_config, &g_lib));
    ucc_lib_config_release(lib_config);

    /* Init ucc context for a specified UCC_TEST_TLS */
    ucc_context_params_t ctx_params = {};
    ctx_params.mask = UCC_CONTEXT_PARAM_FIELD_OOB;
    ctx_params.oob.allgather = oob_allgather;
    ctx_params.oob.req_test = oob_allgather_test;
    ctx_params.oob.req_free = oob_allgather_free;
    ctx_params.oob.coll_info = (void *)MPI_COMM_WORLD;
    ctx_params.oob.n_oob_eps = g_size;
    ctx_params.oob.oob_ep = g_rank;

    ucc_context_config_h ctx_config;
    UCC_CHECK(ucc_context_config_read(g_lib, NULL, &ctx_config));
    /* UCC_CHECK(ucc_context_config_modify(ctx_config, "TLS", &lib_config));     */

    UCC_CHECK(ucc_context_create(g_lib, &ctx_params, ctx_config, &g_ctx));
    ucc_context_config_release(ctx_config);

    g_team = create_ucc_team(MPI_COMM_WORLD, g_ctx);

    return 0;
}

int UCC_finalize() {
    /* Cleanup UCC */
    UCC_CHECK(ucc_team_destroy(g_team));
    UCC_CHECK(ucc_context_destroy(g_ctx));
    UCC_CHECK(ucc_finalize(g_lib));
    return 0;
}

int MPI_Alltoall(const void *sendbuf, int sendcount,
     MPI_Datatype sendtype, void *recvbuf, int recvcount,
     MPI_Datatype recvtype, MPI_Comm comm, cudaStream_t stream) {
    
    ucc_coll_req_h req;
    ucc_coll_args_t args;

    args.mask = 0;
    args.coll_type = UCC_COLL_TYPE_ALLTOALL;
    args.src.info.buffer = (void*) sendbuf;
    args.src.info.count = sendcount;
    args.src.info.datatype = mpi_to_ucc_datatype(sendtype);
    args.src.info.mem_type = UCC_MEMORY_TYPE_CUDA;

    args.dst.info.buffer = recvbuf;
    args.dst.info.count = recvcount;
    args.dst.info.datatype = mpi_to_ucc_datatype(recvtype);
    args.dst.info.mem_type = UCC_MEMORY_TYPE_CUDA;


    UCC_CHECK(ucc_collective_init(&args, &req, g_team));
    UCC_CHECK(ucc_collective_post(req));
    while (UCC_INPROGRESS == ucc_collective_test(req)) {
        UCC_CHECK(ucc_context_progress(g_ctx));
    }
    ucc_collective_finalize(req);   

    return 0;
}
