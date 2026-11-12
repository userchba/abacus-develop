#ifndef ABACUS_SOURCE_LCAO_MODULE_RT_TD_LINALG_H
#define ABACUS_SOURCE_LCAO_MODULE_RT_TD_LINALG_H

#include <complex>

namespace module_rt
{
class DdlaContext;

#ifdef __MPI
//! C = alpha * op(A) * op(B) + beta * C on whole block-cyclic matrices.
//! Runs on LibDDLA when ddla is not null, otherwise on ScaLAPACK (pzgemm).
void td_gemm(const DdlaContext* ddla,
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
void td_geadd(const DdlaContext* ddla,
              const char trans,
              const int m,
              const int n,
              const std::complex<double>& alpha,
              const std::complex<double>* A,
              const int* descA,
              const std::complex<double>& beta,
              std::complex<double>* C,
              const int* descC);
#endif // __MPI

} // namespace module_rt

#endif // ABACUS_SOURCE_LCAO_MODULE_RT_TD_LINALG_H
