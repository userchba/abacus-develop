#include "td_linalg.h"

#include "source_base/module_external/scalapack_connector.h"
#include "source_base/tool_quit.h"

#ifdef __DDLA
#include "source_base/block_cyclic_distribution.h"
#include "source_base/module_external/ddla_context.h"
#include "source_base/module_external/ddla_device.h"

#include <ddla/ddla.h>

#include <cstddef>
#include <exception>
#include <string>
#endif

namespace module_rt
{
#ifdef __MPI
#ifdef __DDLA
namespace
{
using ComplexBlock = ModuleBase::ddla_device::BackendBlock<std::complex<double>>;
using ConstComplexBlock = ModuleBase::ddla_device::BackendBlock<const std::complex<double>>;

void quit_on_ddla_error(const char* where, const std::exception& e)
{
    const std::string message = std::string("LibDDLA call failed: ") + e.what();
    ModuleBase::WARNING_QUIT(where, message);
}
} // namespace
#endif // __DDLA

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
        ConstComplexBlock a(*ddla, A, descA);
        ConstComplexBlock b(*ddla, B, descB);
        ComplexBlock c(*ddla, C, descC);
        ddla::pgemm(ddla->handle(), transa, transb, m, n, k, alpha, a.data(), descA, b.data(), descB, beta, c.data(), descC);
        c.to_host();
    }
    catch (const std::exception& e)
    {
        quit_on_ddla_error("module_rt::td_gemm", e);
    }
#else
    ModuleBase::WARNING_QUIT("module_rt::td_gemm", "ABACUS is built without LibDDLA support!");
#endif // __DDLA
}

void td_geadd(const ModuleBase::DdlaContext* ddla,
              const char trans,
              const int m,
              const int n,
              const std::complex<double>& alpha,
              const std::complex<double>* A,
              const int* descA,
              const std::complex<double>& beta,
              std::complex<double>* C,
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
        ConstComplexBlock a(*ddla, A, descA);
        ComplexBlock c(*ddla, C, descC);
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
        quit_on_ddla_error("module_rt::td_geadd", e);
    }
#else
    ModuleBase::WARNING_QUIT("module_rt::td_geadd", "ABACUS is built without LibDDLA support!");
#endif // __DDLA
}

void td_ddla_getrf(const ModuleBase::DdlaContext& ddla,
                   const BlockCyclicDistribution& dist,
                   std::complex<double>* A,
                   std::vector<int>& ipiv)
{
#ifdef __DDLA
    const int n = dist.nrglb();
    const std::size_t len_ipiv = static_cast<std::size_t>(dist.nrloc()) + static_cast<std::size_t>(dist.nb());
    ipiv.assign(len_ipiv, 0);
    int info = 0;
    try
    {
        ComplexBlock a(ddla, A, dist.desc());
        ddla::pgetrf(ddla.handle(), n, n, a.data(), 1, 1, dist.desc(), ipiv.data(), info);
        a.to_host();
    }
    catch (const std::exception& e)
    {
        quit_on_ddla_error("module_rt::td_ddla_getrf", e);
    }
    if (info != 0)
    {
        const std::string message = "LibDDLA pgetrf found a singular matrix, info = " + std::to_string(info);
        ModuleBase::WARNING_QUIT("module_rt::td_ddla_getrf", message);
    }
#else
    ModuleBase::WARNING_QUIT("module_rt::td_ddla_getrf", "ABACUS is built without LibDDLA support!");
#endif // __DDLA
}

void td_ddla_getrs(const ModuleBase::DdlaContext& ddla,
                   const BlockCyclicDistribution& dist,
                   const char side,
                   const int nrhs,
                   std::complex<double>* A,
                   const std::vector<int>& ipiv,
                   std::complex<double>* B,
                   const int* descB)
{
#ifdef __DDLA
    const int n = dist.nrglb();
    const char no_trans = 'N';
    try
    {
        ComplexBlock a(ddla, A, dist.desc());
        ComplexBlock b(ddla, B, descB);
        ddla::pgetrs(ddla.handle(), side, no_trans, n, nrhs, a.data(), dist.desc(), ipiv.data(), b.data(), descB);
        b.to_host();
    }
    catch (const std::exception& e)
    {
        quit_on_ddla_error("module_rt::td_ddla_getrs", e);
    }
#else
    ModuleBase::WARNING_QUIT("module_rt::td_ddla_getrs", "ABACUS is built without LibDDLA support!");
#endif // __DDLA
}
#endif // __MPI
} // namespace module_rt
