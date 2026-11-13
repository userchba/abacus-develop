#ifndef ABACUS_SOURCE_LCAO_MODULE_RT_DDLA_CONTEXT_H
#define ABACUS_SOURCE_LCAO_MODULE_RT_DDLA_CONTEXT_H

#ifdef __DDLA
#include <ddla/ddla_handle_t.h>

class BlockCyclicDistribution;

namespace module_rt
{

//! Owns one LibDDLA handle bound to the process grid of a block-cyclic
//! distribution. Every propagation step and k-point that uses the same grid
//! reuses this handle instead of creating a new one per call.
class DdlaContext
{
  public:
    DdlaContext() = default;
    //! Release the handle; collective over the bound grid.
    ~DdlaContext();

    DdlaContext(const DdlaContext&) = delete;
    DdlaContext& operator=(const DdlaContext&) = delete;

    //! Bind a CPU (use_gpu false) or GPU handle to the BLACS grid of dist;
    //! collective over its communicator. Binding again to the same BLACS
    //! context and backend keeps the existing handle. A GPU handle uses the
    //! device ABACUS assigns to the rank (node-local rank modulo device count)
    //! when the grid spans MPI_COMM_WORLD.
    void bind(const BlockCyclicDistribution& dist, const bool use_gpu);

    //! Destroy the handle; collective over the bound grid. Safe when unbound.
    void release();

    //! Whether a handle is bound.
    bool is_bound() const
    {
        return handle_ != nullptr;
    }

    //! Whether the bound handle runs on the GPU; LibDDLA calls then take device memory.
    bool on_gpu() const
    {
        return on_gpu_;
    }

    //! BLACS context of the bound grid, or -1 when unbound.
    int ictxt() const
    {
        return ictxt_;
    }

    //! Handle for LibDDLA calls on the bound grid; quits when unbound.
    const ddla::DdlaHandle_t& handle() const;

  private:
    ddla::DdlaHandle_t handle_ = nullptr;
    int ictxt_ = -1;
    bool on_gpu_ = false;
};

} // namespace module_rt

#endif // __DDLA

#endif // ABACUS_SOURCE_LCAO_MODULE_RT_DDLA_CONTEXT_H
