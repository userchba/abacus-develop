#ifndef ABACUS_SOURCE_BASE_MODULE_EXTERNAL_DDLA_LINALG_H
#define ABACUS_SOURCE_BASE_MODULE_EXTERNAL_DDLA_LINALG_H

#include <memory>
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

//! ddla when it runs on a GPU, nullptr (ScaLAPACK) for a CPU handle or none.
//! For call sites where only the GPU backend was measured to be faster than
//! ScaLAPACK.
const DdlaContext* gpu_backend(const DdlaContext* ddla);

#ifdef __MPI
//! Explicit lifetime of a local block in the selected backend. A CPU/native
//! matrix borrows host storage; a GPU matrix uploads once. Intermediates have
//! zero-initialized owned storage. Context and its grid must outlive every matrix.
//! Host edits require refresh(), and host consumers require copy_to_host().
//! These operations never implicitly cache matrices by their host address.
template <typename T>
class ResidentMatrix
{
  public:
    ResidentMatrix(const DdlaContext* context, const T* input, const int* desc);
    ResidentMatrix(const DdlaContext* context, T* output, const int* desc, bool copy_in);
    ResidentMatrix(const DdlaContext* context, const int* desc);
    ~ResidentMatrix();
    ResidentMatrix(ResidentMatrix&&) noexcept;
    ResidentMatrix& operator=(ResidentMatrix&&) noexcept;
    ResidentMatrix(const ResidentMatrix&) = delete;
    ResidentMatrix& operator=(const ResidentMatrix&) = delete;

    const DdlaContext* context() const;
    const int* desc() const;
    const T* data() const;
    T* data();
    void refresh(const T* input);
    void copy_to_host(T* output) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

//! In-place LibDDLA factorization with reusable backend pivots. The matrix
//! and context must outlive this object. CPU and GPU handles are supported;
//! the native null-context solve remains available through gesv().
template <typename T>
class ResidentLU
{
  public:
    explicit ResidentLU(ResidentMatrix<T>& matrix);
    ~ResidentLU();
    ResidentLU(const ResidentLU&) = delete;
    ResidentLU& operator=(const ResidentLU&) = delete;
    int info() const;
    void solve(char side, int nrhs, ResidentMatrix<T>& rhs) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

//! Backend-memory operations: no application-level upload or readback. All
//! operands must belong to the same context (nullptr selects native ScaLAPACK).
template <typename T>
void gemm(char transa, char transb, int m, int n, int k,
          const typename Scalar<T>::type& alpha, const ResidentMatrix<T>& A,
          const ResidentMatrix<T>& B, const typename Scalar<T>::type& beta,
          ResidentMatrix<T>& C);
template <typename T>
void gemm(char transa, char transb, int m, int n, int k,
          const typename Scalar<T>::type& alpha, const ResidentMatrix<T>& A, int ia, int ja,
          const ResidentMatrix<T>& B, int ib, int jb, const typename Scalar<T>::type& beta,
          ResidentMatrix<T>& C, int ic, int jc);
template <typename T>
void geadd(char trans, int m, int n, const typename Scalar<T>::type& alpha,
           const ResidentMatrix<T>& A, const typename Scalar<T>::type& beta, ResidentMatrix<T>& C);
template <typename T>
int gesv(int n, int nrhs, ResidentMatrix<T>& A, ResidentMatrix<T>& B);
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

//! C(ic:, jc:) = alpha * op(A(ia:, ja:)) * op(B(ib:, jb:)) + beta * C(ic:, jc:)
//! on submatrices (PxGEMM convention with one-based offsets).
//! Runs on LibDDLA when ddla is not null, otherwise on ScaLAPACK; LibDDLA
//! multiplies whole matrices, so an operand with an offset is first copied
//! into a whole matrix with the same blocks on the same grid.
template <typename T>
void gemm(const DdlaContext* ddla,
          const char transa,
          const char transb,
          const int m,
          const int n,
          const int k,
          const typename Scalar<T>::type& alpha,
          const T* A,
          const int ia,
          const int ja,
          const int* descA,
          const T* B,
          const int ib,
          const int jb,
          const int* descB,
          const typename Scalar<T>::type& beta,
          T* C,
          const int ic,
          const int jc,
          const int* descC);

//! C = beta * C + alpha * op(A) with op 'T' (PxTRANU) or 'C' (PxTRANC).
//! Runs on LibDDLA (as geadd) when ddla is not null, otherwise on ScaLAPACK
//! PxTRANU / PxTRANC (PxTRAN for real T).
template <typename T>
void tran(const DdlaContext* ddla,
          const char trans,
          const int m,
          const int n,
          const typename Scalar<T>::type& alpha,
          const T* A,
          const int* descA,
          const typename Scalar<T>::type& beta,
          T* C,
          const int* descC);

//! Solve A X = B with an LU factorization (PxGESV); A (n x n) is overwritten
//! by its factors and B (n x nrhs) by X. Returns the factorization's info
//! (0 on success, > 0 when A is singular), the same on every rank.
//! Runs on LibDDLA when ddla is not null, otherwise on ScaLAPACK.
template <typename T>
int gesv(const DdlaContext* ddla, const int n, const int nrhs, T* A, const int* descA, T* B, const int* descB);

//! Solve A X = B for a Hermitian positive-definite A with a Cholesky
//! factorization (PxPOTRF + PxPOTRS) that reads the uplo triangle of A; A is
//! overwritten by the factor and B (n x nrhs) by X. Returns the factorization's
//! info (0 on success, > 0 when A is not positive definite; B is then left
//! unchanged), the same on every rank.
//! Runs on LibDDLA when ddla is not null, otherwise on ScaLAPACK.
template <typename T>
int posv(const DdlaContext* ddla,
         const char uplo,
         const int n,
         const int nrhs,
         T* A,
         const int* descA,
         T* B,
         const int* descB);

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
