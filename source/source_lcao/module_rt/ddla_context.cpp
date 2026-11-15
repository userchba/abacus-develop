#include "ddla_context.h"

#ifdef __DDLA
#include "source_base/block_cyclic_distribution.h"
#include "source_base/tool_quit.h"

#include <mpi.h>

#include <stdexcept>
#include <string>

namespace module_rt
{

namespace
{

// LibDDLA gives the k-th MPI process of a node the GPU k % (GPUs of the
// node). With NCCL/RCCL communication two processes must not share a GPU:
// the collectives hang instead of failing. Collective over comm; returns an
// error message, empty when every process has a GPU of its own.
std::string gpu_sharing_error(const ddla::DdlaHandle_t& handle, const MPI_Comm comm)
{
    if (ddla::ddlaGetCommunication(handle) != ddla::DdlaCommunication::CCL)
    {
        return std::string();
    }
    int gpus = 0;
    ddla::DDLA_CHECK(ddla::ddlaGetDeviceCount(handle, &gpus));
    MPI_Comm node = MPI_COMM_NULL;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
    int node_processes = 0;
    MPI_Comm_size(node, &node_processes);
    MPI_Comm_free(&node);

    // Largest number of processes and smallest number of GPUs on any node.
    int counts[2] = {node_processes, -gpus};
    MPI_Allreduce(MPI_IN_PLACE, counts, 2, MPI_INT, MPI_MAX, comm);
    const int max_processes = counts[0];
    const int min_gpus = -counts[1];
    if (max_processes <= min_gpus)
    {
        return std::string();
    }
    return "LibDDLA on GPU needs one GPU per MPI process, but a node runs " + std::to_string(max_processes)
           + " MPI processes and has only " + std::to_string(min_gpus)
           + " visible GPUs.\n Please run at most one MPI process per GPU on each node.";
}

} // namespace

DdlaContext::~DdlaContext()
{
    release();
}

void DdlaContext::bind(const BlockCyclicDistribution& dist, const bool use_gpu)
{
    const int ictxt = dist.ictxt();
    if (ictxt < 0)
    {
        ModuleBase::WARNING_QUIT("DdlaContext::bind", "the block-cyclic distribution has no BLACS grid");
    }
    if (handle_ != nullptr && ictxt == ictxt_ && use_gpu == on_gpu_)
    {
        return;
    }
    release();

    const MPI_Comm comm = dist.comm();
    const int nprows = dist.nrproc();
    const int npcols = dist.ncproc();
    // BlockCyclicDistribution numbers its BLACS grid in row-major order.
    const char major = 'R';
    const ddla::DdlaBackend backend = use_gpu ? ddla::DdlaBackend::GPU : ddla::DdlaBackend::CPU;
    std::string sharing_error;
    try
    {
        ddla::DDLA_CHECK(ddla::ddlaInit(handle_, backend));
        if (use_gpu)
        {
            sharing_error = gpu_sharing_error(handle_, comm);
        }
        if (sharing_error.empty())
        {
            ddla::DDLA_CHECK(ddla::ddlaSet(handle_, comm, nprows, npcols, major));
        }
    }
    catch (const std::exception& e)
    {
        const std::string message = std::string("LibDDLA handle setup failed: ") + e.what();
        ModuleBase::WARNING_QUIT("DdlaContext::bind", message);
    }
    if (!sharing_error.empty())
    {
        ModuleBase::WARNING_QUIT("DdlaContext::bind", sharing_error);
    }
    ictxt_ = ictxt;
    on_gpu_ = use_gpu;
}

void DdlaContext::release()
{
    if (handle_ == nullptr)
    {
        return;
    }
    ddla::ddlaDestroy(handle_);
    handle_ = nullptr;
    ictxt_ = -1;
    on_gpu_ = false;
}

const ddla::DdlaHandle_t& DdlaContext::handle() const
{
    if (handle_ == nullptr)
    {
        ModuleBase::WARNING_QUIT("DdlaContext::handle", "no LibDDLA handle is bound");
    }
    return handle_;
}

} // namespace module_rt

#endif // __DDLA
