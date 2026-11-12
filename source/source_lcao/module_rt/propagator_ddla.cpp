#include "propagator.h"

#include "source_base/timer.h"
#include "source_base/tool_quit.h"

#ifdef __DDLA
#include "ddla_context.h"
#include "source_base/block_cyclic_distribution.h"

#include <ddla/ddla.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
#endif

#include <complex>
#include <fstream>

namespace module_rt
{
#ifdef __MPI
void Propagator::compute_propagator_ddla(const int nlocal,
                                         const std::complex<double>* Stmp,
                                         const std::complex<double>* Htmp,
                                         std::complex<double>* U_operator,
                                         const DdlaContext& ddla,
                                         std::ofstream& ofs_running,
                                         const int print_matrix) const
{
    switch (ptype)
    {
    case 0:
        compute_propagator_cn2_ddla(nlocal, Stmp, Htmp, U_operator, ddla, ofs_running, print_matrix);
        break;
    default:
        ModuleBase::WARNING_QUIT("Propagator::compute_propagator_ddla",
                                 "td_la_backend ddla currently supports td_propagator 0 only!");
        break;
    }
}

void Propagator::compute_propagator_cn2_ddla(const int nlocal,
                                             const std::complex<double>* Stmp,
                                             const std::complex<double>* Htmp,
                                             std::complex<double>* U_operator,
                                             const DdlaContext& ddla,
                                             std::ofstream& ofs_running,
                                             const int print_matrix) const
{
#ifdef __DDLA
    ModuleBase::timer::start("Propagator", "compute_propagator_cn2_ddla");

    const BlockCyclicDistribution& dist = this->ParaV->para_2d();
    const int* desc = dist.desc();
    const ddla::DdlaHandle_t& handle = ddla.handle();
    const std::size_t len_local = std::max<std::size_t>(dist.size(), 1);
    std::vector<std::complex<double>> denominator(len_local, std::complex<double>(0.0, 0.0));

    const char no_trans = 'N';
    const char side = 'L';
    const std::complex<double> one(1.0, 0.0);
    const std::complex<double> minus_coef(0.0, -0.25 * this->dt);
    const std::complex<double> plus_coef(0.0, 0.25 * this->dt);
    try
    {
        // Numerator = S - i*dt/4*H is written straight into U_operator.
        ddla::pgeadd<ddla::DdlaBackend::CPU>(handle,
                                             no_trans,
                                             no_trans,
                                             nlocal,
                                             nlocal,
                                             one,
                                             Stmp,
                                             1,
                                             1,
                                             desc,
                                             minus_coef,
                                             Htmp,
                                             1,
                                             1,
                                             desc,
                                             U_operator,
                                             1,
                                             1,
                                             desc);
        // Denominator = S + i*dt/4*H.
        ddla::pgeadd<ddla::DdlaBackend::CPU>(handle,
                                             no_trans,
                                             no_trans,
                                             nlocal,
                                             nlocal,
                                             one,
                                             Stmp,
                                             1,
                                             1,
                                             desc,
                                             plus_coef,
                                             Htmp,
                                             1,
                                             1,
                                             desc,
                                             denominator.data(),
                                             1,
                                             1,
                                             desc);
        // U = Denominator^{-1} * Numerator: one LU solve replaces the explicit
        // inverse and the extra matrix product of the ScaLAPACK path.
        ddla::pgesv<ddla::DdlaBackend::CPU>(handle,
                                            side,
                                            no_trans,
                                            nlocal,
                                            nlocal,
                                            denominator.data(),
                                            desc,
                                            U_operator,
                                            desc);
    }
    catch (const std::exception& e)
    {
        const std::string message = std::string("LibDDLA Crank-Nicolson propagator failed: ") + e.what();
        ModuleBase::WARNING_QUIT("Propagator::compute_propagator_cn2_ddla", message);
    }

    if (print_matrix)
    {
        ofs_running << " U operator (LibDDLA):" << std::endl;
        for (int i = 0; i < dist.nrloc(); i++)
        {
            const int in = i * dist.ncloc();
            for (int j = 0; j < dist.ncloc(); j++)
            {
                double aa = U_operator[in + j].real();
                double bb = U_operator[in + j].imag();
                if (std::abs(aa) < 1e-8)
                {
                    aa = 0.0;
                }
                if (std::abs(bb) < 1e-8)
                {
                    bb = 0.0;
                }
                ofs_running << aa << "+" << bb << "i ";
            }
            ofs_running << std::endl;
        }
    }

    ModuleBase::timer::end("Propagator", "compute_propagator_cn2_ddla");
#else
    ModuleBase::WARNING_QUIT("Propagator::compute_propagator_cn2_ddla", "ABACUS is built without LibDDLA support!");
#endif // __DDLA
}
#endif // __MPI
} // namespace module_rt
