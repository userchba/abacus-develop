#include "zeqlin_solv.h"

#ifdef __MPI
#include <cassert>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

#include "source_base/module_external/ddla_linalg.h"
#include "source_base/timer.h"
#include "source_base/tool_title.h"

namespace LR
{
    /// A <- (A + A^H) / 2 on the 2D layout `pA`. The Cholesky solvers read only the upper
    /// triangle, so any numerical asymmetry of A would otherwise be dropped silently instead of
    /// averaged.
    template <typename T>
    void hermitize_2d(const ModuleBase::DdlaContext* ddla, T* A, const BlockCyclicDistribution& pA)
    {
        const int n = pA.nrglb();
        std::vector<T> AH(pA.size(), T(0));
        const T one(1.0);
        const T zero(0.0);
        ModuleBase::ddla_linalg::tran(ddla, 'C', n, n, one, A, pA.desc(), zero, AH.data(), pA.desc());
        for (std::size_t i = 0; i < AH.size(); ++i)
        {
            A[i] = 0.5 * (A[i] + AH[i]);
        }
    }

    template <typename T>
    void scalapack_linear_solver(const ModuleBase::DdlaContext* ddla, T* A, T* B, const BlockCyclicDistribution& pA, const BlockCyclicDistribution& pB)
    {
        ModuleBase::TITLE("LR", "scalapack_linear_solver");
        ModuleBase::timer::start("LR", "scalapack_linear_solver");
        const int n = pA.nrglb();
        const int nrhs = pB.ncglb();
        assert(pA.ncglb() == n);
        assert(pB.nrglb() == n);
        assert(pA.ictxt() == pB.ictxt());
        assert(pA.nb() == pB.nb());
        const int info = ModuleBase::ddla_linalg::gesv(ddla, n, nrhs, A, pA.desc(), B, pB.desc());
        if (info != 0)
        {
            ModuleBase::timer::end("LR", "scalapack_linear_solver");
            throw std::runtime_error("scalapack_linear_solver: p?gesv failed, info=" + std::to_string(info));
        }
        ModuleBase::timer::end("LR", "scalapack_linear_solver");
    }

    template <typename T>
    void scalapack_cholesky_linear_solver(const ModuleBase::DdlaContext* ddla, T* A, T* B, const BlockCyclicDistribution& pA, const BlockCyclicDistribution& pB)
    {
        ModuleBase::TITLE("LR", "scalapack_cholesky_linear_solver");
        ModuleBase::timer::start("LR", "scalapack_cholesky_linear_solver");
        const int n = pA.nrglb();
        const int nrhs = pB.ncglb();
        assert(pA.ncglb() == n);
        assert(pB.nrglb() == n);
        assert(pA.ictxt() == pB.ictxt());
        assert(pA.nb() == pB.nb());

        hermitize_2d(ddla, A, pA);

        // A = U^H U, U in the upper triangle. The factorization's INFO is global output, so a
        // non-positive-definite A is reported identically on every rank.
        const int info = ModuleBase::ddla_linalg::posv(ddla, 'U', n, nrhs, A, pA.desc(), B, pB.desc());
        if (info != 0)
        {
            std::string message = "scalapack_cholesky_linear_solver: p?potrf/p?potrs failed, info="
                + std::to_string(info);
            if (info > 0)
            {
                message += " -- the matrix is not positive definite; "
                    "use lr_grad_solver=scalapack (LU) instead";
            }
            ModuleBase::timer::end("LR", "scalapack_cholesky_linear_solver");
            throw std::runtime_error(message);
        }
        ModuleBase::timer::end("LR", "scalapack_cholesky_linear_solver");
    }

    template void scalapack_linear_solver<double>(const ModuleBase::DdlaContext*, double*, double*, const BlockCyclicDistribution&, const BlockCyclicDistribution&);
    template void scalapack_linear_solver<std::complex<double>>(const ModuleBase::DdlaContext*, std::complex<double>*, std::complex<double>*,
        const BlockCyclicDistribution&, const BlockCyclicDistribution&);
    template void scalapack_cholesky_linear_solver<double>(const ModuleBase::DdlaContext*, double*, double*, const BlockCyclicDistribution&,
        const BlockCyclicDistribution&);
    template void scalapack_cholesky_linear_solver<std::complex<double>>(const ModuleBase::DdlaContext*, std::complex<double>*,
        std::complex<double>*, const BlockCyclicDistribution&, const BlockCyclicDistribution&);

}
#endif
