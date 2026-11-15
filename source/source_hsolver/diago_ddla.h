#ifndef ABACUS_SOURCE_HSOLVER_DIAGO_DDLA_H
#define ABACUS_SOURCE_HSOLVER_DIAGO_DDLA_H

#ifdef __DDLA
#include "source_base/macros.h"
#include "source_base/matrix_block.h"
#include "source_psi/psi.h"

namespace module_rt
{
class DdlaContext;
}

namespace hsolver
{

//! Solves H(k) C = S(k) C diag(e) for the lowest nbands eigenpairs with the
//! distributed LibDDLA phegvd, on the CPU or the GPU of the bound handle.
//! H, S and the eigenvectors stay block-cyclic on the grid of the handle.
template <typename T>
class DiagoDdla
{
  private:
    using Real = typename GetTypeReal<T>::type;

  public:
    /// @param ddla_in LibDDLA handle bound to the process grid of H and S
    /// @param nlocal_in global dimension of the NAO Hamiltonian
    /// @param nbands_in number of lowest eigenpairs to compute
    DiagoDdla(const module_rt::DdlaContext& ddla_in, const int nlocal_in, const int nbands_in)
        : ddla(ddla_in), nlocal(nlocal_in), nbands(nbands_in)
    {
    }

    /// H and S keep their values. The lowest nbands S-orthonormal eigenvectors
    /// are written to psi in the row and column blocks of H, and the lowest
    /// nbands eigenvalues to eigenvalue_in.
    void diag(ModuleBase::MatrixBlock<T>& h_mat,
              ModuleBase::MatrixBlock<T>& s_mat,
              psi::Psi<T>& psi,
              Real* eigenvalue_in);

  private:
    const module_rt::DdlaContext& ddla;
    const int nlocal;
    const int nbands;
};

} // namespace hsolver

#endif // __DDLA

#endif // ABACUS_SOURCE_HSOLVER_DIAGO_DDLA_H
