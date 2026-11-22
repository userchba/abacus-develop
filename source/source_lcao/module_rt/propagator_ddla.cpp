#include "propagator.h"

#include "source_base/block_cyclic_distribution.h"
#include "source_base/module_external/ddla_linalg.h"
#include "source_base/timer.h"
#include "source_base/tool_quit.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <fstream>
#include <vector>

namespace module_rt
{
#ifdef __MPI
namespace
{
using Complex = std::complex<double>;
using ModuleBase::ddla_linalg::ResidentLU;
using ModuleBase::ddla_linalg::ResidentMatrix;

// All Taylor powers and the sum remain in the selected backend. ETRS reuses
// the same overlap factorization for its two half-step exponentials.
void resident_taylor(const ResidentLU<Complex>& overlap_lu,
                     const ResidentMatrix<Complex>& hamiltonian,
                     const ResidentMatrix<Complex>& identity,
                     const Complex& coefficient,
                     ResidentMatrix<Complex>& result)
{
    const ModuleBase::DdlaContext* context = result.context();
    const int* desc = result.desc();
    const int n = desc[2];
    const Complex one(1.0, 0.0);
    const Complex zero(0.0, 0.0);
    const Complex half(0.5, 0.0);
    const Complex sixth(1.0 / 6.0, 0.0);
    const Complex twenty_fourth(1.0 / 24.0, 0.0);
    ResidentMatrix<Complex> x(context, desc);
    ResidentMatrix<Complex> x2(context, desc);
    ResidentMatrix<Complex> x3(context, desc);
    ResidentMatrix<Complex> x4(context, desc);
    ModuleBase::ddla_linalg::geadd('N', n, n, coefficient, hamiltonian, zero, x);
    overlap_lu.solve('L', n, x);
    ModuleBase::ddla_linalg::gemm('N', 'N', n, n, n, one, x, x, zero, x2);
    ModuleBase::ddla_linalg::gemm('N', 'N', n, n, n, one, x, x2, zero, x3);
    ModuleBase::ddla_linalg::gemm('N', 'N', n, n, n, one, x, x3, zero, x4);
    ModuleBase::ddla_linalg::geadd('N', n, n, one, identity, zero, result);
    ModuleBase::ddla_linalg::geadd('N', n, n, one, x, one, result);
    ModuleBase::ddla_linalg::geadd('N', n, n, half, x2, one, result);
    ModuleBase::ddla_linalg::geadd('N', n, n, sixth, x3, one, result);
    ModuleBase::ddla_linalg::geadd('N', n, n, twenty_fourth, x4, one, result);
}

void resident_exponential(const int method,
                          const BlockCyclicDistribution& dist,
                          const double dt,
                          const Complex* overlap_host,
                          const Complex* hamiltonian_host,
                          const Complex* previous_hamiltonian,
                          Complex* output,
                          const ModuleBase::DdlaContext& context)
{
    const int n = dist.nrglb();
    const int* desc = dist.desc();
    const Complex one(1.0, 0.0);
    const Complex zero(0.0, 0.0);
    std::vector<Complex> identity_host(std::max<std::size_t>(dist.size(), 1), zero);
    for (int col = 0; col < dist.ncloc(); ++col)
    {
        for (int row = 0; row < dist.nrloc(); ++row)
        {
            if (dist.irglb(row) == dist.icglb(col))
            {
                identity_host[col * dist.nrloc() + row] = one;
            }
        }
    }
    const ResidentMatrix<Complex> overlap(&context, overlap_host, desc);
    const ResidentMatrix<Complex> hamiltonian(&context, hamiltonian_host, desc);
    const Complex* identity_ptr = identity_host.data();
    const ResidentMatrix<Complex> identity(&context, identity_ptr, desc);
    ResidentMatrix<Complex> factor(&context, desc);
    ModuleBase::ddla_linalg::geadd('N', n, n, one, overlap, zero, factor);
    const ResidentLU<Complex> overlap_lu(factor);
    if (overlap_lu.info() != 0)
    {
        ModuleBase::WARNING_QUIT("Propagator::compute_propagator_ddla", "Singular overlap matrix");
    }
    ResidentMatrix<Complex> result(&context, output, desc, false);
    if (method == 1)
    {
        const Complex coefficient(0.0, -0.5 * dt);
        resident_taylor(overlap_lu, hamiltonian, identity, coefficient, result);
    }
    else
    {
        const Complex coefficient(0.0, -0.25 * dt);
        const ResidentMatrix<Complex> previous(&context, previous_hamiltonian, desc);
        ResidentMatrix<Complex> first(&context, desc);
        ResidentMatrix<Complex> second(&context, desc);
        resident_taylor(overlap_lu, hamiltonian, identity, coefficient, first);
        resident_taylor(overlap_lu, previous, identity, coefficient, second);
        ModuleBase::ddla_linalg::gemm('N', 'N', n, n, n, one, first, second, zero, result);
    }
    result.copy_to_host(output);
}
} // namespace

void Propagator::compute_propagator_ddla(const int nlocal,
                                         const std::complex<double>* Stmp,
                                         const std::complex<double>* Htmp,
                                         const std::complex<double>* H_laststep,
                                         std::complex<double>* U_operator,
                                         const ModuleBase::DdlaContext& ddla,
                                         std::ofstream& ofs_running,
                                         const int print_matrix) const
{
    // Diagnostic printing consumes the existing host intermediates explicitly.
    // The normal numerical region performs one final readback of U.
    if ((ptype == 1 || ptype == 2) && print_matrix == 0)
    {
        resident_exponential(ptype, this->ParaV->para_2d(), this->dt, Stmp, Htmp, H_laststep, U_operator, ddla);
        return;
    }
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
    using ModuleBase::ddla_linalg::ResidentMatrix;
    const std::complex<double> one(1.0, 0.0);
    const std::complex<double> zero(0.0, 0.0);
    const std::complex<double> minus_coef(0.0, -0.25 * this->dt);
    const std::complex<double> plus_coef(0.0, 0.25 * this->dt);
    const ResidentMatrix<std::complex<double>> overlap(&ddla, Stmp, desc);
    const ResidentMatrix<std::complex<double>> hamiltonian(&ddla, Htmp, desc);
    ResidentMatrix<std::complex<double>> denominator(&ddla, desc);
    ResidentMatrix<std::complex<double>> numerator(&ddla, U_operator, desc, false);
    ModuleBase::ddla_linalg::geadd('N', nlocal, nlocal, one, overlap, zero, numerator);
    ModuleBase::ddla_linalg::geadd('N', nlocal, nlocal, minus_coef, hamiltonian, one, numerator);
    ModuleBase::ddla_linalg::geadd('N', nlocal, nlocal, one, overlap, zero, denominator);
    ModuleBase::ddla_linalg::geadd('N', nlocal, nlocal, plus_coef, hamiltonian, one, denominator);
    const int info = ModuleBase::ddla_linalg::gesv(nlocal, nlocal, denominator, numerator);
    if (info != 0)
    {
        ModuleBase::WARNING_QUIT("Propagator::compute_propagator_cn2_ddla", "Singular CN2 denominator");
    }
    numerator.copy_to_host(U_operator);

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
