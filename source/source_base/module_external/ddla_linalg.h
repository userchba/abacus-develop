#ifndef ABACUS_SOURCE_BASE_MODULE_EXTERNAL_DDLA_LINALG_H
#define ABACUS_SOURCE_BASE_MODULE_EXTERNAL_DDLA_LINALG_H

#include <vector>

class BlockCyclicDistribution;

namespace ModuleBase
{
class DdlaContext;

// Distributed linear algebra on whole block-cyclic host matrices. A call runs
// on LibDDLA when it gets a LibDDLA handle and on ScaLAPACK otherwise; with a
// GPU handle it copies the rank's local blocks to its GPU, runs there and
// copies the result back. T is double or std::complex<double>.
namespace ddla_linalg
{
//! The scalar type of matrices of type T. Scalar parameters use it so that T
//! is deduced from the matrices only and a real alpha works for complex ones.
template <typename T>
struct Scalar
{
    using type = T;
};

#ifdef __MPI
//! C = alpha * op(A) * op(B) + beta * C (PxGEMM convention).
//! Runs on LibDDLA when ddla is not null, otherwise on ScaLAPACK.
template <typename T>
void gemm(const DdlaContext* ddla,
          const char transa,
          const char transb,
          const int m,
          const int n,
          const int k,
          const typename Scalar<T>::type& alpha,
          const T* A,
          const int* descA,
          const T* B,
          const int* descB,
          const typename Scalar<T>::type& beta,
          T* C,
          const int* descC);

//! C = beta * C + alpha * op(A) (PxGEADD convention).
//! Runs on LibDDLA when ddla is not null, otherwise on ScaLAPACK.
template <typename T>
void geadd(const DdlaContext* ddla,
           const char trans,
           const int m,
           const int n,
           const typename Scalar<T>::type& alpha,
           const T* A,
           const int* descA,
           const typename Scalar<T>::type& beta,
           T* C,
           const int* descC);

//! LU-factor the square matrix A of layout dist in place with LibDDLA.
//! ipiv is resized and receives LibDDLA's row-distributed pivots.
template <typename T>
void getrf(const DdlaContext& ddla, const BlockCyclicDistribution& dist, T* A, std::vector<int>& ipiv);

//! Solve A X = B (side 'L') or X A = B (side 'R') with the factors from
//! getrf; B (n x nrhs or nrhs x n) is overwritten by X.
template <typename T>
void getrs(const DdlaContext& ddla,
           const BlockCyclicDistribution& dist,
           const char side,
           const int nrhs,
           T* A,
           const std::vector<int>& ipiv,
           T* B,
           const int* descB);
#endif // __MPI
} // namespace ddla_linalg
} // namespace ModuleBase

#endif // ABACUS_SOURCE_BASE_MODULE_EXTERNAL_DDLA_LINALG_H
