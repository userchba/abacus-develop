/**
 * @file norm_psi.h
 * @brief normalize the wave function
 *  This file originally belonged to file LCAO_evolve.cpp
 */
#ifndef NORM_PSI_H
#define NORM_PSI_H

#include "source_base/module_container/ATen/core/tensor.h" // ct::Tensor
#include "source_basis/module_ao/parallel_orbitals.h"
#include "source_lcao/module_rt/kernels/cublasmp_context.h"

#include <complex>

namespace ModuleBase
{
class DdlaContext;
}

namespace module_rt
{
#ifdef __MPI
/**
 *  @brief turn the band overlap psi^H S psi into the normalization factors
 *
 * Within the leading nband x nband block of the local Eij-layout matrix,
 * replace each diagonal element by 1 / sqrt(Re(diagonal)) and each
 * off-diagonal element by zero.
 *
 * @param[in] pv information of parallel
 * @param[in] nband number of bands
 * @param[in,out] Cij local block in the Eij layout (leading dimension nrloc)
 */
void norm_psi_scaling(const Parallel_Orbitals* pv, const int nband, std::complex<double>* Cij);

/**
 *  @brief normalize the wave function
 *
 * @param[in] pv information of parallel
 * @param[in] nband number of bands
 * @param[in] nlocal number of orbitals
 * @param[in] Stmp overlap matrix
 * @param[in] psi_k psi of this step
 * @param[in] print_matirx print internal matrix or not
 * @param[out] psi_k psi of this step after normalization
 */
void norm_psi(const Parallel_Orbitals* pv,
              const ModuleBase::DdlaContext* ddla,
              const int nband,
              const int nlocal,
              const std::complex<double>* Stmp,
              std::complex<double>* psi_k,
              std::ofstream& ofs_running,
              const int print_matrix);

void norm_psi_tensor(const Parallel_Orbitals* pv,
                     const int nband,
                     const int nlocal,
                     const ct::Tensor& Stmp,
                     ct::Tensor& psi_k,
                     std::ofstream& ofs_running,
                     const int print_matrix,
                     CublasMpResources& cublas_res);

template <typename Device>
void norm_psi_tensor_lapack(const Parallel_Orbitals* pv,
                            const int nband,
                            const int nlocal,
                            const ct::Tensor& Stmp,
                            ct::Tensor& psi_k,
                            std::ofstream& ofs_running,
                            const int print_matrix);

#endif // __MPI
} // namespace module_rt

#endif
