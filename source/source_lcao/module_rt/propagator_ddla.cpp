#include "propagator.h"

#include "source_base/block_cyclic_distribution.h"
#include "source_base/timer.h"
#include "source_base/tool_quit.h"
#include "td_linalg.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <fstream>
#include <vector>

namespace module_rt
{
#ifdef __MPI
void Propagator::compute_propagator_ddla(const int nlocal,
                                         const std::complex<double>* Stmp,
                                         const std::complex<double>* Htmp,
                                         const std::complex<double>* H_laststep,
                                         std::complex<double>* U_operator,
                                         const ModuleBase::DdlaContext& ddla,
                                         std::ofstream& ofs_running,
                                         const int print_matrix) const
{
    const int taylor_tag = 1;
    switch (ptype)
    {
    case 0:
        compute_propagator_cn2_ddla(nlocal, Stmp, Htmp, U_operator, ddla, ofs_running, print_matrix);
        break;
    case 1:
        compute_propagator_taylor(nlocal, Stmp, Htmp, U_operator, ofs_running, print_matrix, taylor_tag, &ddla);
        break;
    case 2:
        compute_propagator_etrs(nlocal, Stmp, Htmp, H_laststep, U_operator, ofs_running, print_matrix, &ddla);
        break;
    default:
        ModuleBase::WARNING_QUIT("Propagator::compute_propagator_ddla", "Method of RT-TDDFT propagator is wrong!");
        break;
    }
}

void Propagator::compute_propagator_cn2_ddla(const int nlocal,
                                             const std::complex<double>* Stmp,
                                             const std::complex<double>* Htmp,
                                             std::complex<double>* U_operator,
                                             const ModuleBase::DdlaContext& ddla,
                                             std::ofstream& ofs_running,
                                             const int print_matrix) const
{
    ModuleBase::timer::start("Propagator", "compute_propagator_cn2_ddla");

    const BlockCyclicDistribution& dist = this->ParaV->para_2d();
    const int* desc = dist.desc();
    const std::size_t len_local = std::max<std::size_t>(dist.size(), 1);
    std::vector<std::complex<double>> denominator(len_local, std::complex<double>(0.0, 0.0));
    std::copy(Stmp, Stmp + dist.size(), denominator.begin());
    std::copy(Stmp, Stmp + dist.size(), U_operator);

    const char no_trans = 'N';
    const char left = 'L';
    const std::complex<double> one(1.0, 0.0);
    const std::complex<double> minus_coef(0.0, -0.25 * this->dt);
    const std::complex<double> plus_coef(0.0, 0.25 * this->dt);

    // Numerator S - i*dt/4*H is formed in U_operator, denominator S + i*dt/4*H.
    td_geadd(&ddla, no_trans, nlocal, nlocal, minus_coef, Htmp, desc, one, U_operator, desc);
    td_geadd(&ddla, no_trans, nlocal, nlocal, plus_coef, Htmp, desc, one, denominator.data(), desc);

    // U = Denominator^{-1} * Numerator: one LU solve replaces the explicit
    // inverse and the extra matrix product of the ScaLAPACK path.
    std::vector<int> ipiv;
    td_ddla_getrf(ddla, dist, denominator.data(), ipiv);
    td_ddla_getrs(ddla, dist, left, nlocal, denominator.data(), ipiv, U_operator, desc);

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
}
#endif // __MPI
} // namespace module_rt
