#ifndef ABACUS_SOURCE_LCAO_MODULE_LR_ZEQLIN_SOLV_H
#define ABACUS_SOURCE_LCAO_MODULE_LR_ZEQLIN_SOLV_H
#include "source_base/block_cyclic_distribution.h"

namespace ModuleBase
{
class DdlaContext;
}

namespace LR
{
#ifdef __MPI
    /// @brief Solve A Z = B with an LU factorization (ScaLAPACK p?gesv, or LibDDLA when `ddla`
    /// is a handle bound to the grid of `pA`), the distributed counterpart of
    /// `LR_Util::lapack_linear_solver`.
    /// @param A [in/out] local block of the n x n matrix on `pA`; overwritten by its LU factors
    /// @param B [in/out] local block of the n x nrhs right-hand side on `pB`; overwritten by Z
    /// @attention `pA` and `pB` must share the BLACS context and the block size.
    template <typename T>
    void scalapack_linear_solver(const ModuleBase::DdlaContext* ddla, T* A, T* B, const BlockCyclicDistribution& pA, const BlockCyclicDistribution& pB);

    /// @brief Solve A Z = B for a Hermitian positive-definite A with a Cholesky factorization
    /// A = U^H U, then the triangular solves (ScaLAPACK p?potrf + p?potrs, or LibDDLA when
    /// `ddla` is not null). A is Hermitized as (A + A^H)/2 first.
    /// Arguments as in `scalapack_linear_solver`; A is overwritten by U.
    /// Throws on every rank if A is not positive definite (p?potrf's INFO is global).
    template <typename T>
    void scalapack_cholesky_linear_solver(const ModuleBase::DdlaContext* ddla, T* A, T* B, const BlockCyclicDistribution& pA, const BlockCyclicDistribution& pB);

#endif
}

#endif // ABACUS_SOURCE_LCAO_MODULE_LR_ZEQLIN_SOLV_H
