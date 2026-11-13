#ifdef __DDLA

#include "../ddla_context.h"
#include "../evolve_psi.h"
#include "source_base/parallel_reduce.h"
#include "source_basis/module_ao/parallel_orbitals.h"

#include <ddla/ddla_config.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <tuple>
#include <vector>

/************************************************
 *  unit test of module_rt::evolve_psi with LibDDLA
 ***********************************************/

/**
 * - Tested function
 *   - evolve_psi with a LibDDLA context (td_propagator 0)
 *     - one full Crank-Nicolson step (half-step H and S, propagator, U*psi,
 *       normalization, band energies) gives the same psi and band energies
 *       as the ScaLAPACK path, on the CPU backend and, when LibDDLA has GPU
 *       support, on the GPU backend, whose step keeps all matrices on the
 *       device (evolve_psi_ddla_device).
 */

namespace
{

using Complex = std::complex<double>;

// Grid-independent pseudo-random entry in [-0.5, 0.5) + i[-0.5, 0.5).
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

// Local column-major block of the Hermitian matrix diag * I + scale * (R + R^H) / 2.
std::vector<Complex> hermitian_local(const BlockCyclicDistribution& dist,
                                     const int seed,
                                     const double diag,
                                     const double scale)
{
    const std::size_t len = std::max<std::size_t>(dist.size(), 1);
    std::vector<Complex> local(len, Complex(0.0, 0.0));
    for (int jl = 0; jl < dist.ncloc(); ++jl)
    {
        for (int il = 0; il < dist.nrloc(); ++il)
        {
            const int irglb = dist.irglb(il);
            const int icglb = dist.icglb(jl);
            const Complex upper = test_entry(seed, irglb, icglb);
            const Complex lower = std::conj(test_entry(seed, icglb, irglb));
            const std::size_t idx = static_cast<std::size_t>(jl) * dist.nrloc() + il;
            local[idx] = 0.5 * scale * (upper + lower);
            if (irglb == icglb)
            {
                local[idx] += diag;
            }
        }
    }
    return local;
}

// Hamiltonian that hands evolve_psi the local blocks of H(t+dt) and S(t+dt).
class BlockHamiltonian : public hamilt::Hamilt<Complex>
{
  public:
    std::vector<Complex> h;
    std::vector<Complex> s;
    int nrow = 0;
    int ncol = 0;

    void matrix(hamilt::MatrixBlock<Complex>& hk, hamilt::MatrixBlock<Complex>& sk) override
    {
        hk.p = h.data();
        sk.p = s.data();
        hk.row = nrow;
        sk.row = nrow;
        hk.col = ncol;
        sk.col = ncol;
    }
};

// Run the backends LibDDLA was built with (false: CPU, true: GPU).
#if DDLA_HAS_CPU && DDLA_HAS_GPU
const auto ddla_backends = testing::Values(false, true);
#elif DDLA_HAS_GPU
const auto ddla_backends = testing::Values(true);
#else
const auto ddla_backends = testing::Values(false);
#endif

} // namespace

// Parameters: (nlocal, nband, block size, swap grid), GPU backend.
class EvolvePsiDdlaTest : public testing::TestWithParam<std::tuple<std::tuple<int, int, int, bool>, bool>>
{
  protected:
    std::ofstream log;
};

TEST_P(EvolvePsiDdlaTest, CrankNicolsonStepMatchesScalapack)
{
    const std::tuple<int, int, int, bool>& layout = std::get<0>(GetParam());
    const bool use_gpu = std::get<1>(GetParam());
    const int nlocal = std::get<0>(layout);
    const int nband = std::get<1>(layout);
    const int nb = std::get<2>(layout);
    const bool swap_grid = std::get<3>(layout);

    Parallel_Orbitals pv;
    pv.para_2d().init(nlocal, nlocal, nb, MPI_COMM_WORLD, swap_grid);
    const BlockCyclicDistribution& dist = pv.para_2d();
    int ncol_bands = 0;
    for (int jl = 0; jl < dist.ncloc(); ++jl)
    {
        if (dist.icglb(jl) < nband)
        {
            ++ncol_bands;
        }
    }
    pv.ncol_bands = ncol_bands;
    pv.nrow_bands = dist.nrloc();
    pv.nloc_wfc = static_cast<long>(ncol_bands) * dist.nrloc();
    pv.set_desc_wfc_Eij(nlocal, nband, dist.nrloc());
    // Every rank must own part of psi, as ABACUS requires in production runs.
    ASSERT_GT(pv.nloc_wfc, 0);

    BlockHamiltonian hamiltonian;
    hamiltonian.nrow = dist.nrloc();
    hamiltonian.ncol = dist.ncloc();
    const double s_scale = 0.2 / std::sqrt(static_cast<double>(nlocal));
    hamiltonian.h = hermitian_local(dist, 21, 0.0, 1.0);
    hamiltonian.s = hermitian_local(dist, 22, 1.0, s_scale);
    std::vector<Complex> h_last = hermitian_local(dist, 23, 0.0, 1.0);
    std::vector<Complex> s_last = hermitian_local(dist, 24, 1.0, s_scale);

    const std::size_t len_psi = static_cast<std::size_t>(pv.nloc_wfc);
    std::vector<Complex> psi_last(len_psi, Complex(0.0, 0.0));
    for (int jl = 0; jl < ncol_bands; ++jl)
    {
        for (int il = 0; il < dist.nrloc(); ++il)
        {
            const std::size_t idx = static_cast<std::size_t>(jl) * dist.nrloc() + il;
            psi_last[idx] = test_entry(25, dist.irglb(il), dist.icglb(jl));
        }
    }

    const module_rt::EijOutputInfo output_info{-1.0, 0, 0, 0, 0, 0};
    const int propagator = 0;
    const int print_matrix = 0;
    const bool use_td_moving_gauge = false;

    std::vector<Complex> psi_ref(len_psi, Complex(0.0, 0.0));
    std::vector<double> ekb_ref(nband, 0.0);
    const module_rt::DdlaContext* scalapack = nullptr;
    module_rt::evolve_psi(nband, nlocal, &pv, &hamiltonian, psi_ref.data(), psi_last.data(), h_last.data(),
                          s_last.data(), nullptr, use_td_moving_gauge, ekb_ref.data(), propagator, scalapack, log,
                          print_matrix, output_info);

    module_rt::DdlaContext ddla;
    ddla.bind(dist, use_gpu);
    std::vector<Complex> psi_ddla(len_psi, Complex(0.0, 0.0));
    std::vector<double> ekb_ddla(nband, 0.0);
    module_rt::evolve_psi(nband, nlocal, &pv, &hamiltonian, psi_ddla.data(), psi_last.data(), h_last.data(),
                          s_last.data(), nullptr, use_td_moving_gauge, ekb_ddla.data(), propagator, &ddla, log,
                          print_matrix, output_info);

    double psi_norm = 0.0;
    double psi_diff = 0.0;
    for (std::size_t i = 0; i < len_psi; ++i)
    {
        psi_norm = std::max(psi_norm, std::abs(psi_ref[i]));
        psi_diff = std::max(psi_diff, std::abs(psi_ref[i] - psi_ddla[i]));
    }
    Parallel_Reduce::reduce_max(psi_norm);
    Parallel_Reduce::reduce_max(psi_diff);
    ASSERT_GT(psi_norm, 1.0e-3);
    EXPECT_LT(psi_diff, 1.0e-11 * nlocal) << "gpu=" << use_gpu << " nlocal=" << nlocal << " nband=" << nband
                                          << " grid=" << dist.nrproc() << "x" << dist.ncproc();

    double ekb_scale = 0.0;
    for (int ib = 0; ib < nband; ++ib)
    {
        ekb_scale = std::max(ekb_scale, std::abs(ekb_ref[ib]));
    }
    ASSERT_GT(ekb_scale, 1.0e-3);
    for (int ib = 0; ib < nband; ++ib)
    {
        EXPECT_NEAR(ekb_ddla[ib], ekb_ref[ib], 1.0e-11 * nlocal) << "band " << ib << " gpu=" << use_gpu;
    }
}

INSTANTIATE_TEST_SUITE_P(Layouts,
                         EvolvePsiDdlaTest,
                         testing::Combine(testing::Values(std::make_tuple(40, 24, 4, false),
                                                          std::make_tuple(40, 24, 4, true),
                                                          std::make_tuple(37, 13, 4, false),
                                                          std::make_tuple(64, 48, 8, true)),
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
