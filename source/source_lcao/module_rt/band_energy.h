/**
 * @file band_energy.h
 * @brief compute band energy ekb
 *  This file originally belonged to file LCAO_evolve.cpp
 */
#ifndef BANDENERGY_H
#define BANDENERGY_H

#include "source_base/module_container/ATen/core/tensor.h" // ct::Tensor
#include "source_basis/module_ao/parallel_orbitals.h"
#include "source_io/module_energy/td_eij_io.h"
#include "source_lcao/module_rt/kernels/cublasmp_context.h"

#include <complex>

namespace module_rt
{
class DdlaContext;

using ModuleIO::EijOutputInfo;

#ifdef __MPI
/**
 *  @brief extract the band energies from Eij = psi^H H psi
 *
 * Prints Eij when output_info.threshold is not negative, then collects the
 * diagonal of the distributed Eij into ekb on every rank.
 *
 * @param[in] pv information of parallel
 * @param[in] nband number of bands
 * @param[in] eij local block in the Eij layout (leading dimension nrloc)
 * @param[out] ekb band energy
 */
void ekb_from_eij(const Parallel_Orbitals* pv,
                  const int nband,
                  std::complex<double>* eij,
                  double* ekb,
                  std::ofstream& ofs_running,
                  const EijOutputInfo& output_info);

/**
 *  @brief compute band energy ekb <psi_i|H|psi_i>
 *
 * @param[in] pv information of parallel
 * @param[in] nband number of bands
 * @param[in] nlocal number of orbitals
 * @param[in] Htmp Hamiltonian
 * @param[in] psi_k psi of this step
 * @param[out] ekb band energy
 */
void compute_ekb(const Parallel_Orbitals* pv,
                 const DdlaContext* ddla,
                 const int nband,
                 const int nlocal,
                 const std::complex<double>* Htmp,
                 const std::complex<double>* psi_k,
                 double* ekb,
                 std::ofstream& ofs_running,
                 const EijOutputInfo& output_info);

void compute_ekb_tensor(const Parallel_Orbitals* pv,
                        const int nband,
                        const int nlocal,
                        const ct::Tensor& Htmp,
                        const ct::Tensor& psi_k,
                        ct::Tensor& ekb,
                        std::ofstream& ofs_running,
                        CublasMpResources& cublas_res);

template <typename Device>
void compute_ekb_tensor_lapack(const Parallel_Orbitals* pv,
                               const int nband,
                               const int nlocal,
                               const ct::Tensor& Htmp,
                               const ct::Tensor& psi_k,
                               ct::Tensor& ekb,
                               std::ofstream& ofs_running,
                               const EijOutputInfo& output_info);
#endif // __MPI
} // namespace module_rt
#endif
