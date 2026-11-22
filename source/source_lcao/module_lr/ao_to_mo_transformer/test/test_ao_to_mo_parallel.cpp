#include <gtest/gtest.h>
#include "../ao_to_mo.h"
#include "../cvcx.h"
#include "../../dm_trans/dm_trans.h"
#include "../../dm_trans/dm_diff.h"
#include "../../utils/lr_util.h"
#include <algorithm>
#include <complex>
#include <vector>
#ifdef __DDLA
#include "source_base/module_external/ddla_context.h"
#include <ddla/ddla_config.h>
#endif

/************************************************
 *  unit test of the distributed LR kernels
 ***********************************************/

/**
 * - Tested functions
 *   - ao_to_mo_pblas, CVCX_occ_pblas, CVCX_virt_pblas, cal_dm_trans_pblas,
 *     cal_dm_diff_pblas against the serial BLAS versions, with an occupied
 *     window that splits KS blocks and ranks without local blocks.
 *   - each on ScaLAPACK (null context) and, in a LibDDLA build, on a LibDDLA
 *     handle bound to the grid: CPU, and GPU when LibDDLA has a GPU backend.
 */

#ifdef __MPI
namespace
{
// Backends to run: 0 ScaLAPACK, 1 LibDDLA CPU, 2 LibDDLA GPU.
std::vector<int> backends()
{
    std::vector<int> list = {0};
#ifdef __DDLA
    list.push_back(1);
#if DDLA_HAS_GPU
    list.push_back(2);
#endif
#endif
    return list;
}

template<typename T> T coefficient(const int ao, const int band)
{
    return T(0.001 * (ao + band + 1));
}
template<> std::complex<double> coefficient(const int ao, const int band)
{
    return std::complex<double>(0.001 * (ao + band + 1), 0.0007 * (ao - band));
}
double conjugate(const double value)
{
    return value;
}
std::complex<double> conjugate(const std::complex<double> value)
{
    return std::conj(value);
}
template <typename T> void check_empty_blocks(const int naos, const int nocc, const int nvirt, const int block, const int backend)
{
    // The virtual window is smaller than the KS block; its first band is
    // also inside a block rather than at a block boundary.
    BlockCyclicDistribution pao;
    BlockCyclicDistribution pc;
    BlockCyclicDistribution px;
    LR_Util::setup_2d_division(pao, block, naos, naos);
    LR_Util::setup_2d_division(pc, block, naos, nocc + nvirt, pao.ictxt());
    LR_Util::setup_2d_division(px, block, nvirt, nocc, pao.ictxt());
    const ModuleBase::DdlaContext* ddla = nullptr;
#ifdef __DDLA
    ModuleBase::DdlaContext context;
    if (backend > 0)
    {
        context.bind(pao, backend == 2);
        ddla = &context;
    }
#else
    EXPECT_EQ(backend, 0);
#endif
    const int csize = std::max<int>(1, pc.size());
    const int xsize = std::max<int>(1, px.size());
    std::vector<T> coefficients(csize, T(0));
    std::vector<T> result(xsize, T(0));
    for (int col = 0; col < pc.ncloc(); ++col)
    {
        const int band = pc.icglb(col);
        for (int row = 0; row < pc.nrloc(); ++row)
        {
            const int ao = pc.irglb(row);
            coefficients[col * pc.nrloc() + row] = coefficient<T>(ao, band);
        }
    }
    psi::Psi<T> c(coefficients.data(), 1, pc.ncloc(), pc.nrloc(), true);
    auto identity = LR_Util::newTensor<T>({pao.ncloc(), pao.nrloc()});
    identity.zero();
    for (int col = 0; col < pao.ncloc(); ++col)
    {
        for (int row = 0; row < pao.nrloc(); ++row)
        {
            if (pao.irglb(row) == pao.icglb(col))
            {
                identity.template data<T>()[col * pao.nrloc() + row] = T(1);
            }
        }
    }
    std::vector<T> amplitudes(xsize, T(0));
    std::vector<T> full_amplitudes(nvirt * nocc);
    psi::Psi<T> full_c(1, nocc + nvirt, naos, naos, true);
    for (int band = 0; band < nocc + nvirt; ++band)
    {
        for (int ao = 0; ao < naos; ++ao)
        {
            full_c(0, band, ao) = coefficient<T>(ao, band);
        }
    }
    for (int occ = 0; occ < nocc; ++occ)
    {
        for (int virt = 0; virt < nvirt; ++virt)
        {
            full_amplitudes[occ * nvirt + virt] = T(0.002 * (occ + virt + 1));
        }
    }
    for (int col = 0; col < px.ncloc(); ++col)
    {
        for (int row = 0; row < px.nrloc(); ++row)
        {
            const int occ = px.icglb(col);
            const int virt = px.irglb(row);
            amplitudes[col * px.nrloc() + row] = full_amplitudes[occ * nvirt + virt];
        }
    }
    auto full_identity = LR_Util::newTensor<T>({naos, naos});
    full_identity.zero();
    for (int ao = 0; ao < naos; ++ao)
    {
        full_identity.template data<T>()[ao * naos + ao] = T(1);
    }
    const std::vector<container::Tensor> full_potential = {full_identity};
    const auto spin_view = LR_Util::get_psi_spin(c, 0, 1);
    EXPECT_EQ(spin_view.get_nbands(), c.get_nbands());
    EXPECT_EQ(spin_view.get_nbasis(), c.get_nbasis());
    const std::vector<container::Tensor> potential = {identity};
    const T factor(1);
    T* const local_result = px.size() == 0 ? nullptr : result.data();
    const T* const local_amplitudes = px.size() == 0 ? nullptr : amplitudes.data();
    LR::ao_to_mo_pblas(ddla, potential, pao, c, pc, naos, nocc, nvirt, px,
        local_result, false, LR_Util::MO_TYPE::VO, factor);
    for (int col = 0; col < px.ncloc(); ++col)
    {
        const int occ = px.icglb(col);
        for (int row = 0; row < px.nrloc(); ++row)
        {
            const int virt = nocc + px.irglb(row);
            T expected(0);
            for (int ao = 0; ao < naos; ++ao)
            {
                expected += conjugate(coefficient<T>(ao, virt)) * coefficient<T>(ao, occ);
            }
            EXPECT_NEAR(std::abs(result[col * px.nrloc() + row] - T(expected)), 0, 1e-9);
        }
    }
    std::vector<T> full_result(nvirt * nocc);
    for (const bool occupied : {true, false})
    {
        if (occupied)
        {
            LR::CVCX_occ_pblas(ddla, potential, pao, c, pc, local_amplitudes, px,
                naos, nocc, nvirt, local_result, false, factor);
            LR::CVCX_occ_blas(full_potential, full_c, full_amplitudes.data(),
                naos, nocc, nvirt, full_result.data(), false, factor);
        }
        else
        {
            LR::CVCX_virt_pblas(ddla, potential, pao, c, pc, local_amplitudes, px,
                naos, nocc, nvirt, local_result, false, factor);
            LR::CVCX_virt_blas(full_potential, full_c, full_amplitudes.data(),
                naos, nocc, nvirt, full_result.data(), false, factor);
        }
        for (int col = 0; col < px.ncloc(); ++col)
        {
            for (int row = 0; row < px.nrloc(); ++row)
            {
                const int global = px.icglb(col) * nvirt + px.irglb(row);
                const int local = col * px.nrloc() + row;
                EXPECT_NEAR(std::abs(result[local] - full_result[global]), 0, 1e-9);
            }
        }
    }
    for (const bool transition : {true, false})
    {
        std::vector<container::Tensor> distributed;
        std::vector<container::Tensor> serial;
        if (transition)
        {
            distributed = LR::cal_dm_trans_pblas(ddla, local_amplitudes, px, c, pc,
                naos, nocc, nvirt, pao, factor, LR_Util::MO_TYPE::VO);
            serial = LR::cal_dm_trans_blas(full_amplitudes.data(), full_c,
                nocc, nvirt, factor, LR_Util::MO_TYPE::VO);
        }
        else
        {
            distributed = LR::cal_dm_diff_pblas(ddla, local_amplitudes, px, c, pc,
                naos, nocc, nvirt, pao, false, 1);
            serial = LR::cal_dm_diff_blas(full_amplitudes.data(), full_c,
                naos, nocc, nvirt, false, 1);
        }
        for (int col = 0; col < pao.ncloc(); ++col)
        {
            for (int row = 0; row < pao.nrloc(); ++row)
            {
                const int global = pao.icglb(col) * naos + pao.irglb(row);
                const int local = col * pao.nrloc() + row;
                const T difference = distributed[0].template data<T>()[local] - serial[0].template data<T>()[global];
                EXPECT_NEAR(std::abs(difference), 0, 1e-9);
            }
        }
    }
}
}
TEST(AOToMO, EmptyLocalBlocksReal)
{
    for (const int backend : backends())
    {
        SCOPED_TRACE("backend " + std::to_string(backend));
        check_empty_blocks<double>(16, 10, 3, 32, backend);
        check_empty_blocks<double>(64, 20, 3, 32, backend);
        check_empty_blocks<double>(512, 250, 10, 32, backend);
        // LR on the `lr` path distributes with block size 1.
        check_empty_blocks<double>(37, 7, 5, 1, backend);
    }
}
TEST(AOToMO, EmptyLocalBlocksComplex)
{
    for (const int backend : backends())
    {
        SCOPED_TRACE("backend " + std::to_string(backend));
        check_empty_blocks<std::complex<double>>(16, 10, 3, 32, backend);
        check_empty_blocks<std::complex<double>>(64, 20, 3, 32, backend);
        check_empty_blocks<std::complex<double>>(512, 250, 10, 32, backend);
        check_empty_blocks<std::complex<double>>(37, 7, 5, 1, backend);
    }
}
#endif
