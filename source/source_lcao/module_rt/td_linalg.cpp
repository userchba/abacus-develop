#include "td_linalg.h"

#include "source_base/module_external/scalapack_connector.h"
#include "source_base/tool_quit.h"

#ifdef __DDLA
#include "ddla_context.h"

#include <ddla/ddla.h>

#include <stdexcept>
#include <string>
#endif

namespace module_rt
{
#ifdef __MPI
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
        ddla::pgemm<ddla::DdlaBackend::CPU>(ddla->handle(),
                                            transa,
                                            transb,
                                            m,
                                            n,
                                            k,
                                            alpha,
                                            A,
                                            descA,
                                            B,
                                            descB,
                                            beta,
                                            C,
                                            descC);
    }
    catch (const std::exception& e)
    {
        const std::string message = std::string("LibDDLA pgemm failed: ") + e.what();
        ModuleBase::WARNING_QUIT("module_rt::td_gemm", message);
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
        ddla::pgeadd<ddla::DdlaBackend::CPU>(ddla->handle(),
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
    }
    catch (const std::exception& e)
    {
        const std::string message = std::string("LibDDLA pgeadd failed: ") + e.what();
        ModuleBase::WARNING_QUIT("module_rt::td_geadd", message);
    }
#else
    ModuleBase::WARNING_QUIT("module_rt::td_geadd", "ABACUS is built without LibDDLA support!");
#endif // __DDLA
}
#endif // __MPI
} // namespace module_rt
