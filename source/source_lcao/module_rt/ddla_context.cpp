#include "ddla_context.h"

#ifdef __DDLA
#include "source_base/block_cyclic_distribution.h"
#include "source_base/tool_quit.h"

#include <stdexcept>
#include <string>

namespace module_rt
{

DdlaContext::~DdlaContext()
{
    release();
}

void DdlaContext::bind(const BlockCyclicDistribution& dist)
{
    const int ictxt = dist.ictxt();
    if (ictxt < 0)
    {
        ModuleBase::WARNING_QUIT("DdlaContext::bind", "the block-cyclic distribution has no BLACS grid");
    }
    if (handle_ != nullptr && ictxt == ictxt_)
    {
        return;
    }
    release();

    const MPI_Comm comm = dist.comm();
    const int nprows = dist.nrproc();
    const int npcols = dist.ncproc();
    // BlockCyclicDistribution numbers its BLACS grid in row-major order.
    const char major = 'R';
    try
    {
        ddla::ddlaInit(handle_, ddla::DdlaBackend::CPU);
        ddla::ddlaSet(handle_, comm, nprows, npcols, major);
    }
    catch (const std::exception& e)
    {
        const std::string message = std::string("LibDDLA handle setup failed: ") + e.what();
        ModuleBase::WARNING_QUIT("DdlaContext::bind", message);
    }
    ictxt_ = ictxt;
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
