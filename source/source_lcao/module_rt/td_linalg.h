#ifndef ABACUS_SOURCE_LCAO_MODULE_RT_TD_LINALG_H
#define ABACUS_SOURCE_LCAO_MODULE_RT_TD_LINALG_H

#include <complex>
#include <vector>

class BlockCyclicDistribution;

namespace ModuleBase
{
class DdlaContext;
}

namespace module_rt
{
// Distributed linear algebra for LCAO RT-TDDFT on host matrices. With a GPU
// LibDDLA context, each call copies the rank's local blocks to its GPU, runs
// there and copies the result back.

#ifdef __MPI
//! C = alpha * op(A) * op(B) + beta * C on whole block-cyclic matrices.
//! Runs on LibDDLA when ddla is not null, otherwise on ScaLAPACK (pzgemm).
void td_gemm(const ModuleBase::DdlaContext* ddla,
             const char transa,
             const char transb,
             const int m,
             const int n,
             const int k,
             const std::complex<double>& alpha,
             const std::complex<double>* A,
             const int* descA,
             const std::complex<double>* B,
             const int* descB,
             const std::complex<double>& beta,
             std::complex<double>* C,
             const int* descC);

//! C = beta * C + alpha * op(A) on whole block-cyclic matrices (PxGEADD convention).
//! Runs on LibDDLA when ddla is not null, otherwise on ScaLAPACK (pzgeadd).
void td_geadd(const ModuleBase::DdlaContext* ddla,
              const char trans,
              const int m,
              const int n,
              const std::complex<double>& alpha,
              const std::complex<double>* A,
              const int* descA,
              const std::complex<double>& beta,
              std::complex<double>* C,
              const int* descC);

//! LU-factor the square matrix A of layout dist in place with LibDDLA.
//! ipiv is resized and receives LibDDLA's row-distributed pivots.
void td_ddla_getrf(const ModuleBase::DdlaContext& ddla,
                   const BlockCyclicDistribution& dist,
                   std::complex<double>* A,
                   std::vector<int>& ipiv);

//! Solve A X = B (side 'L') or X A = B (side 'R') with the factors from
//! td_ddla_getrf; B (n x nrhs or nrhs x n) is overwritten by X.
void td_ddla_getrs(const ModuleBase::DdlaContext& ddla,
                   const BlockCyclicDistribution& dist,
                   const char side,
                   const int nrhs,
                   std::complex<double>* A,
                   const std::vector<int>& ipiv,
                   std::complex<double>* B,
                   const int* descB);
#endif // __MPI

} // namespace module_rt

#endif // ABACUS_SOURCE_LCAO_MODULE_RT_TD_LINALG_H
