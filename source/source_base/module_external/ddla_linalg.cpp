#include "source_base/module_external/ddla_linalg.h"

#include "source_base/module_external/blacs_connector.h"
#include "source_base/module_external/scalapack_connector.h"
#include "source_base/tool_quit.h"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <vector>

#ifdef __DDLA
#include "source_base/block_cyclic_distribution.h"
#include "source_base/module_external/ddla_context.h"
#include "source_base/module_external/ddla_device.h"

#include <ddla/ddla.h>

#include <exception>
#include <string>
#endif

namespace ModuleBase
{
namespace ddla_linalg
{
#ifdef __MPI
namespace
{
#ifdef __DDLA
template <typename T>
using Block = ddla_device::BackendBlock<T>;

template <typename T>
using ConstBlock = ddla_device::BackendBlock<const T>;

void quit_on_ddla_error(const char* where, const std::exception& e)
{
    const std::string message = std::string("LibDDLA call failed: ") + e.what();
    ModuleBase::WARNING_QUIT(where, message);
}
#endif // __DDLA

void quit_without_ddla(const char* where)
{
    ModuleBase::WARNING_QUIT(where, "ABACUS is built without LibDDLA support!");
}

// Number of rows of the matrix described by desc that this rank stores.
int local_rows(const int* desc)
{
    int nprow = 0;
    int npcol = 0;
    int myrow = 0;
    int mycol = 0;
    Cblacs_gridinfo(desc[1], &nprow, &npcol, &myrow, &mycol);
    return numroc_(&desc[2], &desc[4], &myrow, &desc[6], &nprow);
}

#ifdef __DDLA
// Descriptor of a whole rows x cols matrix with the blocks and the BLACS grid
// of desc, its first block on process (0, 0).
void whole_desc(const int* desc, const int rows, const int cols, int* out)
{
    int nprow = 0;
    int npcol = 0;
    int myrow = 0;
    int mycol = 0;
    Cblacs_gridinfo(desc[1], &nprow, &npcol, &myrow, &mycol);
    const int zero = 0;
    const int nrloc = numroc_(&rows, &desc[4], &myrow, &zero, &nprow);
    const int whole[9] = {1, desc[1], rows, cols, desc[4], desc[5], 0, 0, std::max(nrloc, 1)};
    std::copy(whole, whole + 9, out);
}

// Copy of the rows x cols submatrix of A starting at (ia, ja), as a whole
// matrix with descriptor desc_out in the handle's memory; A is in the
// handle's memory too.
template <typename T>
ddla_device::DeviceMatrixOf<T> extract(const DdlaContext& ddla,
                                       const T* A,
                                       const int ia,
                                       const int ja,
                                       const int* descA,
                                       const int rows,
                                       const int cols,
                                       int* desc_out)
{
    whole_desc(descA, rows, cols, desc_out);
    ddla_device::DeviceMatrixOf<T> sub = ddla_device::allocate<T>(ddla.handle(), desc_out);
    const char no_trans = 'N';
    const T one(1.0);
    const T zero(0.0);
    ddla::pgeadd(ddla.handle(),
                 no_trans,
                 no_trans,
                 rows,
                 cols,
                 one,
                 A,
                 ia,
                 ja,
                 descA,
                 zero,
                 sub.data(),
                 1,
                 1,
                 desc_out,
                 sub.data(),
                 1,
                 1,
                 desc_out);
    return sub;
}
#endif // __DDLA
} // namespace

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
          const int* descC)
{
    if (ddla == nullptr)
    {
        ScalapackConnector::gemm(transa, transb, m, n, k, alpha, A, 1, 1, descA, B, 1, 1, descB, beta, C, 1, 1, descC);
        return;
    }
#ifdef __DDLA
    try
    {
        ConstBlock<T> a(*ddla, A, descA);
        ConstBlock<T> b(*ddla, B, descB);
        // With beta = 0 the old content of C is not read, so a GPU handle
        // does not copy it to the GPU, unless the local block has padding
        // rows: copying the block back would then overwrite them.
        const bool read_c = !(beta == T(0.0)) || ddla_device::has_padding_rows(descC);
        Block<T> c(*ddla, C, descC, read_c);
        ddla::pgemm(ddla->handle(), transa, transb, m, n, k, alpha, a.data(), descA, b.data(), descB, beta, c.data(), descC);
        c.to_host();
    }
    catch (const std::exception& e)
    {
        quit_on_ddla_error("ModuleBase::ddla_linalg::gemm", e);
    }
#else
    quit_without_ddla("ModuleBase::ddla_linalg::gemm");
#endif // __DDLA
}

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
          const int* descC)
{
    if (ddla == nullptr)
    {
        ScalapackConnector::gemm(transa, transb, m, n, k, alpha, A, ia, ja, descA, B, ib, jb, descB, beta, C, ic, jc,
                                 descC);
        return;
    }
#ifdef __DDLA
    if (m == 0 || n == 0)
    {
        return;
    }
    try
    {
        const ddla::DdlaHandle_t& handle = ddla->handle();
        ConstBlock<T> a(*ddla, A, descA);
        ConstBlock<T> b(*ddla, B, descB);
        // Operands with an offset are copied into whole matrices of their
        // stored shape (before op() is applied).
        const T* a_data = a.data();
        const int* a_desc = descA;
        int a_sub_desc[9];
        ddla_device::DeviceMatrixOf<T> a_sub(handle);
        if ((ia != 1 || ja != 1) && k > 0)
        {
            const int rows = transa == 'N' ? m : k;
            const int cols = transa == 'N' ? k : m;
            a_sub = extract(*ddla, a.data(), ia, ja, descA, rows, cols, a_sub_desc);
            a_data = a_sub.data();
            a_desc = a_sub_desc;
        }
        const T* b_data = b.data();
        const int* b_desc = descB;
        int b_sub_desc[9];
        ddla_device::DeviceMatrixOf<T> b_sub(handle);
        if ((ib != 1 || jb != 1) && k > 0)
        {
            const int rows = transb == 'N' ? k : n;
            const int cols = transb == 'N' ? n : k;
            b_sub = extract(*ddla, b.data(), ib, jb, descB, rows, cols, b_sub_desc);
            b_data = b_sub.data();
            b_desc = b_sub_desc;
        }
        // C is read when beta != 0, and copied in whole when only a part of it
        // is written or its local block has padding rows (copying the block
        // back would overwrite them).
        const bool read_c = !(beta == T(0.0));
        const bool c_offset = ic != 1 || jc != 1;
        Block<T> c(*ddla, C, descC, read_c || c_offset || ddla_device::has_padding_rows(descC));
        T* c_data = c.data();
        const int* c_desc = descC;
        int c_sub_desc[9];
        ddla_device::DeviceMatrixOf<T> c_sub(handle);
        if (c_offset)
        {
            if (read_c)
            {
                c_sub = extract(*ddla, static_cast<const T*>(c.data()), ic, jc, descC, m, n, c_sub_desc);
            }
            else
            {
                whole_desc(descC, m, n, c_sub_desc);
                c_sub = ddla_device::allocate<T>(handle, c_sub_desc);
            }
            c_data = c_sub.data();
            c_desc = c_sub_desc;
        }
        ddla::pgemm(handle, transa, transb, m, n, k, alpha, a_data, a_desc, b_data, b_desc, beta, c_data, c_desc);
        if (c_offset)
        {
            const char no_trans = 'N';
            const T one(1.0);
            const T zero(0.0);
            ddla::pgeadd(handle,
                         no_trans,
                         no_trans,
                         m,
                         n,
                         one,
                         c_sub.data(),
                         1,
                         1,
                         c_sub_desc,
                         zero,
                         c_sub.data(),
                         1,
                         1,
                         c_sub_desc,
                         c.data(),
                         ic,
                         jc,
                         descC);
        }
        c.to_host();
    }
    catch (const std::exception& e)
    {
        quit_on_ddla_error("ModuleBase::ddla_linalg::gemm", e);
    }
#else
    quit_without_ddla("ModuleBase::ddla_linalg::gemm");
#endif // __DDLA
}

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
          const int* descC)
{
    if (trans != 'T' && trans != 'C')
    {
        ModuleBase::WARNING_QUIT("ModuleBase::ddla_linalg::tran", "trans must be 'T' or 'C'");
    }
    if (ddla == nullptr)
    {
        if (trans == 'T')
        {
            ScalapackConnector::tranu(m, n, alpha, A, 1, 1, descA, beta, C, 1, 1, descC);
        }
        else
        {
            ScalapackConnector::tranc(m, n, alpha, A, 1, 1, descA, beta, C, 1, 1, descC);
        }
        return;
    }
    geadd(ddla, trans, m, n, alpha, A, descA, beta, C, descC);
}

template <typename T>
int gesv(const DdlaContext* ddla, const int n, const int nrhs, T* A, const int* descA, T* B, const int* descB)
{
    int info = 0;
    // PxGESV and LibDDLA pgetrf need LOCr(M_A) + MB_A pivot entries.
    std::vector<int> ipiv(static_cast<std::size_t>(local_rows(descA)) + static_cast<std::size_t>(descA[4]), 0);
    if (ddla == nullptr)
    {
        ScalapackConnector::gesv(n, nrhs, A, 1, 1, descA, ipiv.data(), B, 1, 1, descB, &info);
        return info;
    }
#ifdef __DDLA
    try
    {
        const ddla::DdlaHandle_t& handle = ddla->handle();
        Block<T> a(*ddla, A, descA);
        Block<T> b(*ddla, B, descB);
        ddla::pgetrf(handle, n, n, a.data(), 1, 1, descA, ipiv.data(), info);
        if (info == 0)
        {
            const char left = 'L';
            const char no_trans = 'N';
            ddla::pgetrs(handle, left, no_trans, n, nrhs, a.data(), descA, ipiv.data(), b.data(), descB);
            b.to_host();
        }
        a.to_host();
    }
    catch (const std::exception& e)
    {
        quit_on_ddla_error("ModuleBase::ddla_linalg::gesv", e);
    }
#else
    quit_without_ddla("ModuleBase::ddla_linalg::gesv");
#endif // __DDLA
    return info;
}

template <typename T>
int posv(const DdlaContext* ddla,
         const char uplo,
         const int n,
         const int nrhs,
         T* A,
         const int* descA,
         T* B,
         const int* descB)
{
    int info = 0;
    if (ddla == nullptr)
    {
        int desc_a[9];
        std::copy(descA, descA + 9, desc_a);
        info = ScalapackConnector::potrf(uplo, n, A, desc_a);
        if (info == 0)
        {
            ScalapackConnector::potrs(uplo, n, nrhs, A, 1, 1, descA, B, 1, 1, descB, &info);
        }
        return info;
    }
#ifdef __DDLA
    try
    {
        const ddla::DdlaHandle_t& handle = ddla->handle();
        Block<T> a(*ddla, A, descA);
        Block<T> b(*ddla, B, descB);
        ddla::ppotrf(handle, uplo, n, a.data(), 1, 1, descA, info);
        if (info == 0)
        {
            const char left = 'L';
            const char no_trans = 'N';
            const T* factor = a.data();
            ddla::ppotrs(handle, left, uplo, no_trans, n, nrhs, factor, descA, b.data(), descB);
            b.to_host();
        }
        a.to_host();
    }
    catch (const std::exception& e)
    {
        quit_on_ddla_error("ModuleBase::ddla_linalg::posv", e);
    }
#else
    quit_without_ddla("ModuleBase::ddla_linalg::posv");
#endif // __DDLA
    return info;
}

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
           const int* descC)
{
    if (ddla == nullptr)
    {
        ScalapackConnector::geadd(trans, m, n, alpha, A, 1, 1, descA, beta, C, 1, 1, descC);
        return;
    }
#ifdef __DDLA
    // LibDDLA computes C = alpha * op(A) + beta * op(B); B = C reproduces PxGEADD.
    const char no_trans = 'N';
    try
    {
        ConstBlock<T> a(*ddla, A, descA);
        Block<T> c(*ddla, C, descC);
        ddla::pgeadd(ddla->handle(),
                     trans,
                     no_trans,
                     m,
                     n,
                     alpha,
                     a.data(),
                     1,
                     1,
                     descA,
                     beta,
                     c.data(),
                     1,
                     1,
                     descC,
                     c.data(),
                     1,
                     1,
                     descC);
        c.to_host();
    }
    catch (const std::exception& e)
    {
        quit_on_ddla_error("ModuleBase::ddla_linalg::geadd", e);
    }
#else
    quit_without_ddla("ModuleBase::ddla_linalg::geadd");
#endif // __DDLA
}

template <typename T>
void getrf(const DdlaContext& ddla, const BlockCyclicDistribution& dist, T* A, std::vector<int>& ipiv)
{
#ifdef __DDLA
    const int n = dist.nrglb();
    const std::size_t len_ipiv = static_cast<std::size_t>(dist.nrloc()) + static_cast<std::size_t>(dist.nb());
    ipiv.assign(len_ipiv, 0);
    int info = 0;
    try
    {
        Block<T> a(ddla, A, dist.desc());
        ddla::pgetrf(ddla.handle(), n, n, a.data(), 1, 1, dist.desc(), ipiv.data(), info);
        a.to_host();
    }
    catch (const std::exception& e)
    {
        quit_on_ddla_error("ModuleBase::ddla_linalg::getrf", e);
    }
    if (info != 0)
    {
        const std::string message = "LibDDLA pgetrf found a singular matrix, info = " + std::to_string(info);
        ModuleBase::WARNING_QUIT("ModuleBase::ddla_linalg::getrf", message);
    }
#else
    (void)ddla;
    (void)dist;
    (void)A;
    (void)ipiv;
    quit_without_ddla("ModuleBase::ddla_linalg::getrf");
#endif // __DDLA
}

template <typename T>
void getrs(const DdlaContext& ddla,
           const BlockCyclicDistribution& dist,
           const char side,
           const int nrhs,
           T* A,
           const std::vector<int>& ipiv,
           T* B,
           const int* descB)
{
#ifdef __DDLA
    const int n = dist.nrglb();
    const char no_trans = 'N';
    try
    {
        Block<T> a(ddla, A, dist.desc());
        Block<T> b(ddla, B, descB);
        ddla::pgetrs(ddla.handle(), side, no_trans, n, nrhs, a.data(), dist.desc(), ipiv.data(), b.data(), descB);
        b.to_host();
    }
    catch (const std::exception& e)
    {
        quit_on_ddla_error("ModuleBase::ddla_linalg::getrs", e);
    }
#else
    (void)ddla;
    (void)dist;
    (void)side;
    (void)nrhs;
    (void)A;
    (void)ipiv;
    (void)B;
    (void)descB;
    quit_without_ddla("ModuleBase::ddla_linalg::getrs");
#endif // __DDLA
}

template void gemm<double>(const DdlaContext*, char, char, int, int, int, const double&, const double*, const int*,
                           const double*, const int*, const double&, double*, const int*);
template void gemm<std::complex<double>>(const DdlaContext*, char, char, int, int, int, const std::complex<double>&,
                                         const std::complex<double>*, const int*, const std::complex<double>*,
                                         const int*, const std::complex<double>&, std::complex<double>*, const int*);
template void gemm<double>(const DdlaContext*, char, char, int, int, int, const double&, const double*, int, int,
                           const int*, const double*, int, int, const int*, const double&, double*, int, int,
                           const int*);
template void gemm<std::complex<double>>(const DdlaContext*, char, char, int, int, int, const std::complex<double>&,
                                         const std::complex<double>*, int, int, const int*,
                                         const std::complex<double>*, int, int, const int*,
                                         const std::complex<double>&, std::complex<double>*, int, int, const int*);
template void tran<double>(const DdlaContext*, char, int, int, const double&, const double*, const int*, const double&,
                           double*, const int*);
template void tran<std::complex<double>>(const DdlaContext*, char, int, int, const std::complex<double>&,
                                         const std::complex<double>*, const int*, const std::complex<double>&,
                                         std::complex<double>*, const int*);
template int gesv<double>(const DdlaContext*, int, int, double*, const int*, double*, const int*);
template int gesv<std::complex<double>>(const DdlaContext*, int, int, std::complex<double>*, const int*,
                                        std::complex<double>*, const int*);
template int posv<double>(const DdlaContext*, char, int, int, double*, const int*, double*, const int*);
template int posv<std::complex<double>>(const DdlaContext*, char, int, int, std::complex<double>*, const int*,
                                        std::complex<double>*, const int*);
template void geadd<double>(const DdlaContext*, char, int, int, const double&, const double*, const int*,
                            const double&, double*, const int*);
template void geadd<std::complex<double>>(const DdlaContext*, char, int, int, const std::complex<double>&,
                                          const std::complex<double>*, const int*, const std::complex<double>&,
                                          std::complex<double>*, const int*);
template void getrf<double>(const DdlaContext&, const BlockCyclicDistribution&, double*, std::vector<int>&);
template void getrf<std::complex<double>>(const DdlaContext&, const BlockCyclicDistribution&, std::complex<double>*,
                                          std::vector<int>&);
template void getrs<double>(const DdlaContext&, const BlockCyclicDistribution&, char, int, double*,
                            const std::vector<int>&, double*, const int*);
template void getrs<std::complex<double>>(const DdlaContext&, const BlockCyclicDistribution&, char, int,
                                          std::complex<double>*, const std::vector<int>&, std::complex<double>*,
                                          const int*);
#endif // __MPI
} // namespace ddla_linalg
} // namespace ModuleBase
