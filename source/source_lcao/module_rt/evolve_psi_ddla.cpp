#include "evolve_psi_ddla.h"

#include "norm_psi.h"
#include "source_base/constants.h"
#include "source_base/timer.h"
#include "source_base/tool_quit.h"

#ifdef __DDLA
#include "ddla_context.h"
#include "td_ddla_device.h"
#endif

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace module_rt
{
#ifdef __MPI
bool evolve_psi_ddla_device_available(const DdlaContext* ddla, const int propagator)
{
#ifdef __DDLA
    return ddla != nullptr && ddla->on_gpu() && propagator == 0;
#else
    return false;
#endif
}

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
                            const EijOutputInfo& output_info)
{
#ifdef __DDLA
    ModuleBase::timer::start("module_rt", "evolve_psi_ddla_device");

    using ddla_device::DeviceMatrix;
    const int* desc = pv->para_2d().desc();
    const int* desc_wfc = pv->desc_wfc;
    const int* desc_eij = pv->desc_Eij;
    const char no_trans = 'N';
    const char conj_trans = 'C';
    const char left = 'L';
    const std::complex<double> one(1.0, 0.0);
    const std::complex<double> zero(0.0, 0.0);
    const std::complex<double> half(0.5, 0.0);
    const double dt_au = td_dt / ModuleBase::AU_to_FS;
    const std::complex<double> minus_coef(0.0, -0.25 * dt_au);
    const std::complex<double> plus_coef(0.0, 0.25 * dt_au);

    // Host copy of the nband x nband matrices (normalization factors, Eij).
    const std::size_t len_eij = std::max<std::size_t>(pv->para_2d().size(), 1);
    std::vector<std::complex<double>> eij_host(len_eij, zero);

    try
    {
        const ddla::DdlaHandle_t& handle = ddla.handle();

        // H(t+dt) and S(t+dt) stay on the device for the band energy and the normalization.
        const DeviceMatrix d_h = ddla_device::to_device(handle, h_mat, desc);
        const DeviceMatrix d_s = ddla_device::to_device(handle, s_mat, desc);
        DeviceMatrix d_u = ddla_device::zeros(handle, desc);
        {
            DeviceMatrix d_denominator = ddla_device::zeros(handle, desc);
            {
                // (1) H(t+dt/2) and S(t+dt/2)
                DeviceMatrix d_h_mid = ddla_device::zeros(handle, desc);
                DeviceMatrix d_s_mid = ddla_device::zeros(handle, desc);
                {
                    const DeviceMatrix d_h_last = ddla_device::to_device(handle, H_laststep, desc);
                    const DeviceMatrix d_s_last = ddla_device::to_device(handle, S_laststep, desc);
                    ddla::pgeadd(handle, no_trans, no_trans, nlocal, nlocal, half, d_h_last.data(), 1, 1, desc,
                                 half, d_h.data(), 1, 1, desc, d_h_mid.data(), 1, 1, desc);
                    ddla::pgeadd(handle, no_trans, no_trans, nlocal, nlocal, half, d_s_last.data(), 1, 1, desc,
                                 half, d_s.data(), 1, 1, desc, d_s_mid.data(), 1, 1, desc);
                }
                // (2) numerator S - i*dt/4*H in d_u, denominator S + i*dt/4*H
                ddla::pgeadd(handle, no_trans, no_trans, nlocal, nlocal, one, d_s_mid.data(), 1, 1, desc,
                             minus_coef, d_h_mid.data(), 1, 1, desc, d_u.data(), 1, 1, desc);
                ddla::pgeadd(handle, no_trans, no_trans, nlocal, nlocal, one, d_s_mid.data(), 1, 1, desc,
                             plus_coef, d_h_mid.data(), 1, 1, desc, d_denominator.data(), 1, 1, desc);
            }
            // U = denominator^{-1} * numerator
            const std::size_t len_ipiv = static_cast<std::size_t>(pv->para_2d().nrloc())
                                         + static_cast<std::size_t>(pv->para_2d().nb());
            std::vector<int> ipiv(len_ipiv, 0);
            int info = 0;
            ddla::pgetrf(handle, nlocal, nlocal, d_denominator.data(), 1, 1, desc, ipiv.data(), info);
            if (info != 0)
            {
                const std::string message = "pgetrf found a singular matrix, info = " + std::to_string(info);
                throw std::runtime_error(message);
            }
            ddla::pgetrs(handle, left, no_trans, nlocal, nlocal, d_denominator.data(), desc, ipiv.data(),
                         d_u.data(), desc);
        }

        // (3) psi(t+dt) = U * psi(t)
        DeviceMatrix d_psi = ddla_device::zeros(handle, desc_wfc);
        {
            const DeviceMatrix d_psi_last = ddla_device::to_device(handle, psi_k_laststep, desc_wfc);
            ddla::pgemm(handle, no_trans, no_trans, nlocal, nband, nlocal, one, d_u.data(), desc,
                        d_psi_last.data(), desc_wfc, zero, d_psi.data(), desc_wfc);
        }

        // (4) normalize: psi <- psi * diag(1 / sqrt(psi^H S psi))
        DeviceMatrix d_tmp = ddla_device::zeros(handle, desc_wfc);
        DeviceMatrix d_eij = ddla_device::zeros(handle, desc_eij);
        ddla::pgemm(handle, no_trans, no_trans, nlocal, nband, nlocal, one, d_s.data(), desc, d_psi.data(),
                    desc_wfc, zero, d_tmp.data(), desc_wfc);
        ddla::pgemm(handle, conj_trans, no_trans, nband, nband, nlocal, one, d_psi.data(), desc_wfc,
                    d_tmp.data(), desc_wfc, zero, d_eij.data(), desc_eij);
        ddla_device::to_host(handle, eij_host.data(), d_eij, desc_eij);
        norm_psi_scaling(pv, nband, eij_host.data());
        ddla_device::copy_to_device(handle, d_eij, eij_host.data(), desc_eij);
        DeviceMatrix d_psi_norm = ddla_device::zeros(handle, desc_wfc);
        ddla::pgemm(handle, no_trans, no_trans, nlocal, nband, nband, one, d_psi.data(), desc_wfc,
                    d_eij.data(), desc_eij, zero, d_psi_norm.data(), desc_wfc);

        // (5) Eij = psi^H H(t+dt) psi
        ddla::pgemm(handle, no_trans, no_trans, nlocal, nband, nlocal, one, d_h.data(), desc,
                    d_psi_norm.data(), desc_wfc, zero, d_tmp.data(), desc_wfc);
        ddla::pgemm(handle, conj_trans, no_trans, nband, nband, nlocal, one, d_psi_norm.data(), desc_wfc,
                    d_tmp.data(), desc_wfc, zero, d_eij.data(), desc_eij);
        ddla_device::to_host(handle, eij_host.data(), d_eij, desc_eij);
        ddla_device::to_host(handle, psi_k, d_psi_norm, desc_wfc);
    }
    catch (const std::exception& e)
    {
        const std::string message = std::string("LibDDLA GPU propagation failed: ") + e.what();
        ModuleBase::WARNING_QUIT("module_rt::evolve_psi_ddla_device", message);
    }

    ekb_from_eij(pv, nband, eij_host.data(), ekb, ofs_running, output_info);

    ModuleBase::timer::end("module_rt", "evolve_psi_ddla_device");
#else
    ModuleBase::WARNING_QUIT("module_rt::evolve_psi_ddla_device", "ABACUS is built without LibDDLA support!");
#endif
}
#endif // __MPI
} // namespace module_rt
