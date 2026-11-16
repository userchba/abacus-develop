#include "source_base/module_external/ddla_context.h"

#ifdef __DDLA
#include "source_base/block_cyclic_distribution.h"
#include "source_base/module_device/device.h"
#include "source_base/parallel_device.h"
#include "source_base/timer.h"
#include "source_base/tool_quit.h"

#include <stdexcept>
#include <string>

namespace ModuleBase
{

namespace
{

// LibDDLA gives the process with node-local rank k in comm the GPU
// k % (visible GPUs). With NCCL/RCCL communication two processes must not
// share a GPU: the collectives hang instead of failing. Collective over comm,
// so that every process (rank 0 included) reports the error; returns an
// error message, empty when every process has a GPU of its own.
std::string gpu_sharing_error(const ddla::DdlaHandle_t& handle, const MPI_Comm comm)
{
    if (ddla::ddlaGetCommunication(handle) != ddla::DdlaCommunication::CCL)
    {
        return std::string();
    }
    int gpus = 0;
    ddla::DDLA_CHECK(ddla::ddlaGetDeviceCount(handle, &gpus));
    const int node_rank = base_device::information::get_node_rank_with_mpi_shared(comm);
    // Number of processes in comm without a GPU of their own.
    double sharing = (node_rank < gpus) ? 0.0 : 1.0;
    Parallel_Common::reduce_data(&sharing, 1, comm);
    if (sharing == 0.0)
    {
        return std::string();
    }
    const int sharing_processes = static_cast<int>(sharing);
    return "LibDDLA on GPU needs one GPU per MPI process, but " + std::to_string(sharing_processes)
           + " MPI process(es) have no GPU of their own: a node runs more MPI processes than its visible GPUs ("
           + std::to_string(gpus) + " on this node).\n Please run at most one MPI process per GPU on each node.";
}

} // namespace

DdlaContext::~DdlaContext()
{
    release();
}

void DdlaContext::bind(const BlockCyclicDistribution& dist, const bool use_gpu)
{
    ModuleBase::timer::start("DdlaContext", "bind");
    const int ictxt = dist.ictxt();
    if (ictxt < 0)
    {
        ModuleBase::WARNING_QUIT("DdlaContext::bind", "the block-cyclic distribution has no BLACS grid");
    }
    if (handle_ != nullptr && ictxt == ictxt_ && use_gpu == on_gpu_)
    {
        ModuleBase::timer::end("DdlaContext", "bind");
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
    ModuleBase::timer::end("DdlaContext", "bind");
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

} // namespace ModuleBase

#endif // __DDLA
