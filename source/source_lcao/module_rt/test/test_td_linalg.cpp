#ifdef __DDLA

#include "../ddla_context.h"
#include "../td_linalg.h"
#include "source_base/module_external/scalapack_connector.h"
#include "source_base/parallel_reduce.h"
#include "source_basis/module_ao/parallel_orbitals.h"

#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

/************************************************
 *  unit test of module_rt::td_gemm / td_geadd
 ***********************************************/

/**
 * - Tested functions
 *   - td_gemm, td_geadd with a LibDDLA context
 *     - match the ScaLAPACK path (null context) for the products used by
 *       one Crank-Nicolson step: H*psi, psi^H*(S psi) into the band x band
 *       Eij layout, psi*Cij, and the H/S half-step average, on uneven
 *       blocks, nband < nlocal and swapped grids, on the CPU and (when
 *       available) the GPU backend.
 *   - td_ddla_getrf + td_ddla_getrs
 *     - side 'L' matches ScaLAPACK pzgesv for S X = B with B in the psi
 *       (nlocal x nband) layout;
 *     - side 'R' solves X S = H: the residual X S - H vanishes.
 */

namespace
{

using Complex = std::complex<double>;

Complex test_entry(const int seed, const int irglb, const int icglb)
{
    const std::uint64_t seed64 = static_cast<std::uint64_t>(seed);
    const std::uint64_t row64 = static_cast<std::uint64_t>(irglb);
    const std::uint64_t col64 = static_cast<std::uint64_t>(icglb);
    std::uint64_t state = seed64 * 0x9E3779B97F4A7C15ULL + row64 * 0x100000001B3ULL + col64;
    double parts[2] = {0.0, 0.0};
    for (int k = 0; k < 2; ++k)
    {
        state += 0x9E3779B97F4A7C15ULL;
        std::uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        parts[k] = static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0) - 0.5;
    }
    return Complex(parts[0], parts[1]);
}

// Local column-major block (leading dimension nrloc of the nlocal x nlocal
// layout) of an nrow x ncol matrix sharing that layout's blocks and grid,
// as for ABACUS's desc_wfc and desc_Eij.
std::vector<Complex> fill_local(const BlockCyclicDistribution& dist, const int nrow, const int ncol, const int seed)
{
    const std::size_t len = std::max<std::size_t>(dist.size(), 1);
    std::vector<Complex> local(len, Complex(0.0, 0.0));
    for (int jl = 0; jl < dist.ncloc(); ++jl)
    {
        for (int il = 0; il < dist.nrloc(); ++il)
        {
            const int irglb = dist.irglb(il);
            const int icglb = dist.icglb(jl);
            if (irglb < nrow && icglb < ncol)
            {
                const std::size_t idx = static_cast<std::size_t>(jl) * dist.nrloc() + il;
                local[idx] = test_entry(seed, irglb, icglb);
            }
        }
    }
    return local;
}

double max_abs_diff_all_ranks(const std::vector<Complex>& a, const std::vector<Complex>& b)
{
    double diff = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        diff = std::max(diff, std::abs(a[i] - b[i]));
    }
    Parallel_Reduce::reduce_max(diff);
    return diff;
}

} // namespace

// Run the backends LibDDLA was built with (false: CPU, true: GPU).
#if DDLA_HAS_CPU && DDLA_HAS_GPU
const auto ddla_backends = testing::Values(false, true);
#elif DDLA_HAS_GPU
const auto ddla_backends = testing::Values(true);
#else
const auto ddla_backends = testing::Values(false);
#endif

// Parameters: (nlocal, nband, block size, swap grid), GPU backend.
class TdLinalgTest : public testing::TestWithParam<std::tuple<std::tuple<int, int, int, bool>, bool>>
{
  protected:
    void SetUp() override
    {
        const std::tuple<int, int, int, bool>& layout = std::get<0>(GetParam());
        const bool use_gpu = std::get<1>(GetParam());
        nlocal = std::get<0>(layout);
        nband = std::get<1>(layout);
        const int nb = std::get<2>(layout);
        const bool swap_grid = std::get<3>(layout);
        pv.para_2d().init(nlocal, nlocal, nb, MPI_COMM_WORLD, swap_grid);
        pv.set_desc_wfc_Eij(nlocal, nband, pv.para_2d().nrloc());
        ddla.bind(pv.para_2d(), use_gpu);
        tol = 1.0e-13 * nlocal;
    }

    int nlocal = 0;
    int nband = 0;
    double tol = 0.0;
    Parallel_Orbitals pv;
    module_rt::DdlaContext ddla;
};

TEST_P(TdLinalgTest, GemmMatchesScalapack)
{
    const BlockCyclicDistribution& dist = pv.para_2d();
    const std::vector<Complex> h = fill_local(dist, nlocal, nlocal, 1);
    const std::vector<Complex> psi = fill_local(dist, nlocal, nband, 2);
    const Complex one(1.0, 0.0);
    const Complex zero(0.0, 0.0);
    const module_rt::DdlaContext* scalapack = nullptr;

    // tmp = H * psi
    std::vector<Complex> tmp_ref(h.size(), zero);
    std::vector<Complex> tmp_ddla(h.size(), zero);
    module_rt::td_gemm(scalapack, 'N', 'N', nlocal, nband, nlocal, one, h.data(), dist.desc(), psi.data(),
                       pv.desc_wfc, zero, tmp_ref.data(), pv.desc_wfc);
    module_rt::td_gemm(&ddla, 'N', 'N', nlocal, nband, nlocal, one, h.data(), dist.desc(), psi.data(),
                       pv.desc_wfc, zero, tmp_ddla.data(), pv.desc_wfc);
    EXPECT_LT(max_abs_diff_all_ranks(tmp_ref, tmp_ddla), tol) << "H*psi";

    // eij = psi^H * tmp in the nband x nband Eij layout
    std::vector<Complex> eij_ref(h.size(), zero);
    std::vector<Complex> eij_ddla(h.size(), zero);
    module_rt::td_gemm(scalapack, 'C', 'N', nband, nband, nlocal, one, psi.data(), pv.desc_wfc, tmp_ref.data(),
                       pv.desc_wfc, zero, eij_ref.data(), pv.desc_Eij);
    module_rt::td_gemm(&ddla, 'C', 'N', nband, nband, nlocal, one, psi.data(), pv.desc_wfc, tmp_ref.data(),
                       pv.desc_wfc, zero, eij_ddla.data(), pv.desc_Eij);
    EXPECT_LT(max_abs_diff_all_ranks(eij_ref, eij_ddla), tol) << "psi^H*tmp";

    // psi_new = tmp * eij
    std::vector<Complex> out_ref(h.size(), zero);
    std::vector<Complex> out_ddla(h.size(), zero);
    module_rt::td_gemm(scalapack, 'N', 'N', nlocal, nband, nband, one, tmp_ref.data(), pv.desc_wfc,
                       eij_ref.data(), pv.desc_Eij, zero, out_ref.data(), pv.desc_wfc);
    module_rt::td_gemm(&ddla, 'N', 'N', nlocal, nband, nband, one, tmp_ref.data(), pv.desc_wfc, eij_ref.data(),
                       pv.desc_Eij, zero, out_ddla.data(), pv.desc_wfc);
    EXPECT_LT(max_abs_diff_all_ranks(out_ref, out_ddla), tol * nband) << "tmp*eij";
}

TEST_P(TdLinalgTest, GeaddMatchesScalapack)
{
    const BlockCyclicDistribution& dist = pv.para_2d();
    const std::vector<Complex> h_last = fill_local(dist, nlocal, nlocal, 3);
    std::vector<Complex> h_ref = fill_local(dist, nlocal, nlocal, 4);
    std::vector<Complex> h_ddla = h_ref;
    const Complex half(0.5, 0.0);
    const Complex coef(0.25, -0.75);
    const module_rt::DdlaContext* scalapack = nullptr;

    module_rt::td_geadd(scalapack, 'N', nlocal, nlocal, half, h_last.data(), dist.desc(), coef, h_ref.data(),
                        dist.desc());
    module_rt::td_geadd(&ddla, 'N', nlocal, nlocal, half, h_last.data(), dist.desc(), coef, h_ddla.data(),
                        dist.desc());
    EXPECT_LT(max_abs_diff_all_ranks(h_ref, h_ddla), 1.0e-15) << "geadd";
}

namespace
{

// S = I + scale * (R + R^H) / 2 stays well conditioned for scale < 1.
std::vector<Complex> overlap_local(const BlockCyclicDistribution& dist, const int nlocal, const int seed)
{
    std::vector<Complex> local = fill_local(dist, nlocal, nlocal, seed);
    const double scale = 0.2 / std::sqrt(static_cast<double>(nlocal));
    for (int jl = 0; jl < dist.ncloc(); ++jl)
    {
        for (int il = 0; il < dist.nrloc(); ++il)
        {
            const int irglb = dist.irglb(il);
            const int icglb = dist.icglb(jl);
            const std::size_t idx = static_cast<std::size_t>(jl) * dist.nrloc() + il;
            const Complex lower = std::conj(test_entry(seed, icglb, irglb));
            local[idx] = 0.5 * scale * (local[idx] + lower);
            if (irglb == icglb)
            {
                local[idx] += 1.0;
            }
        }
    }
    return local;
}

} // namespace

TEST_P(TdLinalgTest, GetrsMatchesScalapack)
{
    const BlockCyclicDistribution& dist = pv.para_2d();
    const std::vector<Complex> s = overlap_local(dist, nlocal, 5);
    const std::vector<Complex> h = fill_local(dist, nlocal, nlocal, 6);
    const std::vector<Complex> b = fill_local(dist, nlocal, nband, 7);
    const Complex one(1.0, 0.0);
    const Complex minus_one(-1.0, 0.0);
    const char left = 'L';
    const char right = 'R';

    // Left solve S X = B, as in td_propagator 3, against ScaLAPACK pzgesv.
    std::vector<Complex> s_ref = s;
    std::vector<Complex> x_ref = b;
    std::vector<int> ipiv_ref(static_cast<std::size_t>(dist.nrloc()) + dist.nb(), 0);
    int info = 0;
    ScalapackConnector::gesv(nlocal, nband, s_ref.data(), 1, 1, dist.desc(), ipiv_ref.data(), x_ref.data(), 1, 1,
                             pv.desc_wfc, &info);
    ASSERT_EQ(info, 0);
    std::vector<Complex> s_lu = s;
    std::vector<Complex> x_ddla = b;
    std::vector<int> ipiv;
    module_rt::td_ddla_getrf(ddla, dist, s_lu.data(), ipiv);
    module_rt::td_ddla_getrs(ddla, dist, left, nband, s_lu.data(), ipiv, x_ddla.data(), pv.desc_wfc);
    EXPECT_LT(max_abs_diff_all_ranks(x_ref, x_ddla), tol) << "S X = B";

    // Right solve X S = H with the same factors; check the residual X S - H.
    std::vector<Complex> x_right = h;
    module_rt::td_ddla_getrs(ddla, dist, right, nlocal, s_lu.data(), ipiv, x_right.data(), dist.desc());
    std::vector<Complex> residual = h;
    const module_rt::DdlaContext* scalapack = nullptr;
    module_rt::td_gemm(scalapack, 'N', 'N', nlocal, nlocal, nlocal, one, x_right.data(), dist.desc(), s.data(),
                       dist.desc(), minus_one, residual.data(), dist.desc());
    const std::vector<Complex> zeros(residual.size(), Complex(0.0, 0.0));
    EXPECT_LT(max_abs_diff_all_ranks(residual, zeros), tol) << "X S = H";
}

INSTANTIATE_TEST_SUITE_P(Layouts,
                         TdLinalgTest,
                         testing::Combine(testing::Values(std::make_tuple(4, 2, 1, false),
                                                          std::make_tuple(37, 13, 4, false),
                                                          std::make_tuple(64, 24, 8, false),
                                                          std::make_tuple(50, 40, 8, true)),
                                          ddla_backends));

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}

#endif // __DDLA
