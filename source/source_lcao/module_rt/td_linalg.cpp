#include "td_linalg.h"

#include "source_base/module_external/scalapack_connector.h"
#include "source_base/tool_quit.h"

#ifdef __DDLA
#include "ddla_context.h"
#include "source_base/block_cyclic_distribution.h"
#include "td_ddla_device.h"

#include <ddla/ddla.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#endif

namespace module_rt
{
#ifdef __MPI
#ifdef __DDLA
namespace
{
#if DDLA_HAS_GPU
using ddla_device::DeviceMatrix;
using ddla_device::to_device;
using ddla_device::to_host;
#endif // DDLA_HAS_GPU

void quit_on_ddla_error(const char* where, const std::exception& e)
{
    const std::string message = std::string("LibDDLA call failed: ") + e.what();
    ModuleBase::WARNING_QUIT(where, message);
}
} // namespace
#endif // __DDLA

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
        const ddla::DdlaHandle_t& handle = ddla->handle();
#if DDLA_HAS_GPU
        if (ddla->on_gpu())
        {
            const DeviceMatrix d_a = to_device(handle, A, descA);
            const DeviceMatrix d_b = to_device(handle, B, descB);
            DeviceMatrix d_c = to_device(handle, C, descC);
            ddla::pgemm<ddla::DdlaBackend::GPU>(handle,
                                                transa,
                                                transb,
                                                m,
                                                n,
                                                k,
                                                alpha,
                                                d_a.data(),
                                                descA,
                                                d_b.data(),
                                                descB,
                                                beta,
                                                d_c.data(),
                                                descC);
            to_host(handle, C, d_c, descC);
            return;
        }
#endif // DDLA_HAS_GPU
#if DDLA_HAS_CPU
        ddla::pgemm<ddla::DdlaBackend::CPU>(handle, transa, transb, m, n, k, alpha, A, descA, B, descB, beta, C, descC);
#else
        throw std::runtime_error("LibDDLA is built without CPU support");
#endif // DDLA_HAS_CPU
    }
    catch (const std::exception& e)
    {
        quit_on_ddla_error("module_rt::td_gemm", e);
    }
#else
    ModuleBase::WARNING_QUIT("module_rt::td_gemm", "ABACUS is built without LibDDLA support!");
#endif // __DDLA
}

void td_geadd(const DdlaContext* ddla,
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
        const ddla::DdlaHandle_t& handle = ddla->handle();
#if DDLA_HAS_GPU
        if (ddla->on_gpu())
        {
            const DeviceMatrix d_a = to_device(handle, A, descA);
            DeviceMatrix d_c = to_device(handle, C, descC);
            ddla::pgeadd<ddla::DdlaBackend::GPU>(handle,
                                                 trans,
                                                 no_trans,
                                                 m,
                                                 n,
                                                 alpha,
                                                 d_a.data(),
                                                 1,
                                                 1,
                                                 descA,
                                                 beta,
                                                 d_c.data(),
                                                 1,
                                                 1,
                                                 descC,
                                                 d_c.data(),
                                                 1,
                                                 1,
                                                 descC);
            to_host(handle, C, d_c, descC);
            return;
        }
#endif // DDLA_HAS_GPU
#if DDLA_HAS_CPU
        ddla::pgeadd<ddla::DdlaBackend::CPU>(handle,
                                             trans,
                                             no_trans,
                                             m,
                                             n,
                                             alpha,
                                             A,
                                             1,
                                             1,
                                             descA,
                                             beta,
                                             C,
                                             1,
                                             1,
                                             descC,
                                             C,
                                             1,
                                             1,
                                             descC);
#else
        throw std::runtime_error("LibDDLA is built without CPU support");
#endif // DDLA_HAS_CPU
    }
    catch (const std::exception& e)
    {
        quit_on_ddla_error("module_rt::td_geadd", e);
    }
#else
    ModuleBase::WARNING_QUIT("module_rt::td_geadd", "ABACUS is built without LibDDLA support!");
#endif // __DDLA
}

void td_ddla_getrf(const DdlaContext& ddla,
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
        const ddla::DdlaHandle_t& handle = ddla.handle();
#if DDLA_HAS_GPU
        if (ddla.on_gpu())
        {
            DeviceMatrix d_a = to_device(handle, A, dist.desc());
            ddla::pgetrf<ddla::DdlaBackend::GPU>(handle, n, n, d_a.data(), 1, 1, dist.desc(), ipiv.data(), info);
            to_host(handle, A, d_a, dist.desc());
        }
        else
#endif // DDLA_HAS_GPU
        {
#if DDLA_HAS_CPU
            ddla::pgetrf<ddla::DdlaBackend::CPU>(handle, n, n, A, 1, 1, dist.desc(), ipiv.data(), info);
#else
            throw std::runtime_error("LibDDLA is built without CPU support");
#endif // DDLA_HAS_CPU
        }
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

void td_ddla_getrs(const DdlaContext& ddla,
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
        const ddla::DdlaHandle_t& handle = ddla.handle();
#if DDLA_HAS_GPU
        if (ddla.on_gpu())
        {
            DeviceMatrix d_a = to_device(handle, A, dist.desc());
            DeviceMatrix d_b = to_device(handle, B, descB);
            ddla::pgetrs<ddla::DdlaBackend::GPU>(handle,
                                                 side,
                                                 no_trans,
                                                 n,
                                                 nrhs,
                                                 d_a.data(),
                                                 dist.desc(),
                                                 ipiv.data(),
                                                 d_b.data(),
                                                 descB);
            to_host(handle, B, d_b, descB);
            return;
        }
#endif // DDLA_HAS_GPU
#if DDLA_HAS_CPU
        ddla::pgetrs<ddla::DdlaBackend::CPU>(handle, side, no_trans, n, nrhs, A, dist.desc(), ipiv.data(), B, descB);
#else
        throw std::runtime_error("LibDDLA is built without CPU support");
#endif // DDLA_HAS_CPU
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
