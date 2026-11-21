#ifdef __DDLA

#include "source_base/matrix.h"
#include "source_base/module_external/ddla_context.h"
#include "source_basis/module_ao/parallel_orbitals.h"
#include "source_estate/module_dm/density_matrix.h"
#include "source_estate/module_dm/dm_from_psi.h"
#include "source_psi/psi.h"

#include <ddla/ddla.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <tuple>
#include <vector>

/************************************************
 *  unit test of module_dm::dm_from_psi with LibDDLA
 ***********************************************/

/**
 * - Tested function
 *   - module_dm::dm_from_psi(ParaV, wg, wfc, DM, ddla), Gamma-only (double)
 *     and multi-k (complex, real and complex DM(R) types), for the CPU and/or
 *     GPU LibDDLA backend: every k-point block matches the ScaLAPACK
 *     dm_from_psi() and the wavefunctions are left unchanged. One case has
 *     fewer band blocks than process columns on 4 and 6 processes, so the
 *     layout falls back to block size 1 as in an LCAO run.
 */

namespace
{

struct DmCase
{
    int nlocal;
    int nbands;
    int nb;
    int nk;
};

const std::vector<DmCase> dm_cases = {{97, 40, 4, 2}, {260, 130, 16, 3}, {64, 5, 8, 2}};

// Uniform value in [-1, 1) from a 64-bit key (splitmix64).
double hashed_value(std::uint64_t key)
{
    key += 0x9e3779b97f4a7c15ULL;
    key = (key ^ (key >> 30)) * 0xbf58476d1ce4e5b9ULL;
    key = (key ^ (key >> 27)) * 0x94d049bb133111ebULL;
    key = key ^ (key >> 31);
    return static_cast<double>(key >> 11) / static_cast<double>(1ULL << 52) - 1.0;
}

template <typename T>
T matrix_value(const std::uint64_t key);

template <>
double matrix_value<double>(const std::uint64_t key)
{
    return hashed_value(key);
}

template <>
std::complex<double> matrix_value<std::complex<double>>(const std::uint64_t key)
{
    return {hashed_value(2 * key), hashed_value(2 * key + 1)};
}

// The layout of an LCAO run (LCAO_domain::init_basis_lcao): H/S/DM on
// nlocal x nlocal, psi on nlocal x nbands; block size 1 when the bands have
// fewer blocks than the process grid has columns.
void init_orbitals(Parallel_Orbitals& pv, const DmCase& dc)
{
    std::ofstream ofs;
    int try_nb = pv.para_2d().init(dc.nlocal, dc.nlocal, dc.nb, MPI_COMM_WORLD, false);
    try_nb += pv.set_nloc_wfc_Eij(dc.nbands, ofs, ofs);
    if (try_nb != 0)
    {
        pv.para_2d().set(dc.nlocal, dc.nlocal, 1, pv.para_2d().ictxt());
        ASSERT_EQ(pv.set_nloc_wfc_Eij(dc.nbands, ofs, ofs), 0);
    }
    pv.set_desc_wfc_Eij(dc.nlocal, dc.nbands, pv.para_2d().nrloc());
}

// Wavefunctions of nk k-points, element (ik, iband, iw) from its global indices.
template <typename T>
psi::Psi<T> make_wfc(const Parallel_Orbitals& pv, const DmCase& dc)
{
    const int nrow = pv.para_2d().nrloc();
    psi::Psi<T> wfc(dc.nk, pv.ncol_bands, nrow, nrow, true);
    for (int ik = 0; ik < dc.nk; ++ik)
    {
        wfc.fix_k(ik);
        for (int ib = 0; ib < pv.ncol_bands; ++ib)
        {
            const std::uint64_t iband = pv.para_2d().icglb(ib);
            for (int iw = 0; iw < nrow; ++iw)
            {
                const std::uint64_t ibasis = pv.para_2d().irglb(iw);
                const std::uint64_t key = (ik * 1000003ULL + iband) * 1000003ULL + ibasis;
                wfc.get_pointer()[static_cast<std::size_t>(ib) * nrow + iw] = matrix_value<T>(key);
            }
        }
    }
    wfc.fix_k(0);
    return wfc;
}

// Every element of the wavefunctions, all k-points.
template <typename T>
std::vector<T> all_values(const psi::Psi<T>& wfc)
{
    wfc.fix_k(0);
    const T* begin = wfc.get_pointer();
    return std::vector<T>(begin, begin + wfc.size());
}

// Occupation-like weights: decreasing with the band, different for every k-point.
ModuleBase::matrix make_wg(const DmCase& dc)
{
    ModuleBase::matrix wg(dc.nk, dc.nbands);
    for (int ik = 0; ik < dc.nk; ++ik)
    {
        for (int ib = 0; ib < dc.nbands; ++ib)
        {
            wg(ik, ib) = (2.0 - 2.0 * ib / dc.nbands) / (ik + 1);
        }
    }
    return wg;
}

// Fills every k-point block with a finite nonzero value, so that the test
// checks that LibDDLA overwrites the previous density matrix.
template <typename TK, typename TR>
void fill_dm(module_dm::DensityMatrix<TK, TR>& dm, const int nks, const int size)
{
    for (int ik = 0; ik < nks; ++ik)
    {
        std::fill(dm.get_dmk_ptr(ik), dm.get_dmk_ptr(ik) + size, TK(123.0));
    }
}

// Largest |dm - ref| over all processes, relative to the largest |ref|.
template <typename TK, typename TR>
double relative_difference(const module_dm::DensityMatrix<TK, TR>& dm,
                           const module_dm::DensityMatrix<TK, TR>& ref,
                           const int nks,
                           const int size)
{
    double diff[2] = {0.0, 0.0};
    for (int ik = 0; ik < nks; ++ik)
    {
        const TK* a = dm.get_dmk_ptr(ik);
        const TK* b = ref.get_dmk_ptr(ik);
        for (int i = 0; i < size; ++i)
        {
            diff[0] = std::max(diff[0], std::abs(a[i] - b[i]));
            diff[1] = std::max(diff[1], std::abs(b[i]));
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, diff, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    EXPECT_GT(diff[1], 0.0);
    return diff[0] / diff[1];
}

template <typename TK, typename TR>
void check_case(const DmCase& dc,
                const bool gpu,
                module_dm::DensityMatrix<TK, TR>& dm,
                module_dm::DensityMatrix<TK, TR>& ref,
                const Parallel_Orbitals& pv)
{
    const int size = pv.para_2d().nrloc() * pv.para_2d().ncloc();
    const psi::Psi<TK> wfc = make_wfc<TK>(pv, dc);
    const ModuleBase::matrix wg = make_wg(dc);
    const std::vector<TK> wfc_before = all_values(wfc);

    module_dm::dm_from_psi(&pv, wg, wfc, ref);

    ModuleBase::DdlaContext ddla;
    ddla.bind(pv.para_2d(), gpu);
    fill_dm(dm, dc.nk, size);
    module_dm::dm_from_psi(&pv, wg, wfc, dm, ddla);

    EXPECT_LT(relative_difference(dm, ref, dc.nk, size), 1e-13);
    EXPECT_EQ(all_values(wfc), wfc_before);
}

std::vector<bool> ddla_backends()
{
    std::vector<bool> backends;
    backends.push_back(false);
#if DDLA_HAS_GPU
    backends.push_back(true);
#endif
    return backends;
}

} // namespace

class DmFromPsiDdlaTest : public ::testing::TestWithParam<std::tuple<int, bool>>
{
};

TEST_P(DmFromPsiDdlaTest, GammaOnly)
{
    const DmCase& dc = dm_cases[std::get<0>(GetParam())];
    Parallel_Orbitals pv;
    init_orbitals(pv, dc);
    // Gamma-only: one block per spin channel, nk channels here.
    module_dm::DensityMatrix<double, double> dm(&pv, dc.nk);
    module_dm::DensityMatrix<double, double> ref(&pv, dc.nk);
    check_case(dc, std::get<1>(GetParam()), dm, ref, pv);
}

TEST_P(DmFromPsiDdlaTest, KPointsRealDmR)
{
    const DmCase& dc = dm_cases[std::get<0>(GetParam())];
    Parallel_Orbitals pv;
    init_orbitals(pv, dc);
    const std::vector<ModuleBase::Vector3<double>> kvec(dc.nk, ModuleBase::Vector3<double>(0.0, 0.0, 0.0));
    module_dm::DensityMatrix<std::complex<double>, double> dm(&pv, 1, kvec, dc.nk);
    module_dm::DensityMatrix<std::complex<double>, double> ref(&pv, 1, kvec, dc.nk);
    check_case(dc, std::get<1>(GetParam()), dm, ref, pv);
}

TEST_P(DmFromPsiDdlaTest, KPointsComplexDmR)
{
    const DmCase& dc = dm_cases[std::get<0>(GetParam())];
    Parallel_Orbitals pv;
    init_orbitals(pv, dc);
    const std::vector<ModuleBase::Vector3<double>> kvec(dc.nk, ModuleBase::Vector3<double>(0.0, 0.0, 0.0));
    module_dm::DensityMatrix<std::complex<double>, std::complex<double>> dm(&pv, 1, kvec, dc.nk);
    module_dm::DensityMatrix<std::complex<double>, std::complex<double>> ref(&pv, 1, kvec, dc.nk);
    check_case(dc, std::get<1>(GetParam()), dm, ref, pv);
}

INSTANTIATE_TEST_SUITE_P(Cases,
                         DmFromPsiDdlaTest,
                         ::testing::Combine(::testing::Values(0, 1, 2), ::testing::ValuesIn(ddla_backends())));

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    testing::InitGoogleTest(&argc, argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
    if (rank != 0)
    {
        delete listeners.Release(listeners.default_result_printer());
    }
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}

#endif // __DDLA
