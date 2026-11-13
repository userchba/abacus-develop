#ifndef ABACUS_SOURCE_LCAO_MODULE_RT_EVOLVE_PSI_DDLA_H
#define ABACUS_SOURCE_LCAO_MODULE_RT_EVOLVE_PSI_DDLA_H

#include "source_basis/module_ao/parallel_orbitals.h"
#include "source_lcao/module_rt/band_energy.h"

#include <complex>
#include <fstream>

namespace module_rt
{
class DdlaContext;

#ifdef __MPI
//! Whether evolve_psi_ddla_device can run this step: a GPU LibDDLA context
//! and the Crank-Nicolson propagator (td_propagator 0).
bool evolve_psi_ddla_device_available(const DdlaContext* ddla, const int propagator);

/**
 *  @brief evolve psi by one Crank-Nicolson step with all matrices kept on the GPU
 *
 * Same result as evolve_psi with td_propagator 0. The local blocks of H, S and
 * psi are uploaded once, every distributed operation runs on LibDDLA's GPU
 * backend, and only psi, the nband x nband normalization matrix and the band
 * energy matrix return to the host. Each rank holds at most six local
 * nlocal x nlocal blocks and three local psi blocks on its GPU.
 *
 * @param[in] nband number of bands
 * @param[in] nlocal number of orbitals
 * @param[in] pv information of parallel
 * @param[in] ddla GPU LibDDLA context bound to pv's process grid
 * @param[in] td_dt time step in fs
 * @param[in] h_mat H(t+dt)
 * @param[in] s_mat S(t+dt)
 * @param[in] psi_k_laststep psi of last step
 * @param[in] H_laststep H(t)
 * @param[in] S_laststep S(t)
 * @param[out] psi_k psi of this step
 * @param[out] ekb band energy
 */
void evolve_psi_ddla_device(const int nband,
                            const int nlocal,
                            const Parallel_Orbitals* pv,
                            const DdlaContext& ddla,
                            const double td_dt,
                            const std::complex<double>* h_mat,
                            const std::complex<double>* s_mat,
                            const std::complex<double>* psi_k_laststep,
                            const std::complex<double>* H_laststep,
                            const std::complex<double>* S_laststep,
                            std::complex<double>* psi_k,
                            double* ekb,
                            std::ofstream& ofs_running,
                            const EijOutputInfo& output_info);
#endif // __MPI

} // namespace module_rt

#endif // ABACUS_SOURCE_LCAO_MODULE_RT_EVOLVE_PSI_DDLA_H
