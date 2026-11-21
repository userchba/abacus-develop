#include "source_base/module_external/ddla_linalg.h"

#include "source_base/module_external/scalapack_connector.h"
#include "source_base/tool_quit.h"

#include <complex>

#ifdef __DDLA
#include "source_base/block_cyclic_distribution.h"
#include "source_base/module_external/ddla_context.h"
#include "source_base/module_external/ddla_device.h"

#include <ddla/ddla.h>

#include <cstddef>
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
