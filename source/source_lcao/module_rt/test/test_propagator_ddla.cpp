#ifdef __DDLA

#include "../ddla_context.h"
#include "../propagator.h"
#include "source_base/constants.h"
#include "source_base/parallel_reduce.h"
#include "source_basis/module_ao/parallel_orbitals.h"

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
 *  unit test of Propagator::compute_propagator_ddla
 ***********************************************/

/**
 * - Tested function
 *   - Propagator::compute_propagator_ddla (td_propagator 0, 1, 2)
 *     - matches the ScaLAPACK Crank-Nicolson, 4th-order Taylor and ETRS
 *       propagators on ABACUS block-cyclic layouts with uneven blocks and
 *       swapped grids.
 */

namespace
{

// Grid-independent pseudo-random entry in [-0.5, 0.5) + i[-0.5, 0.5).
std::complex<double> test_entry(const int seed, const int irglb, const int icglb)
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
    return std::complex<double>(parts[0], parts[1]);
}

// Hermitian test matrix: diag * I + scale * (R + R^H) / 2.
std::complex<double> hermitian_entry(const int seed,
                                     const double diag,
                                     const double scale,
                                     const int irglb,
                                     const int icglb)
{
    const std::complex<double> upper = test_entry(seed, irglb, icglb);
    const std::complex<double> lower = std::conj(test_entry(seed, icglb, irglb));
    std::complex<double> value = 0.5 * scale * (upper + lower);
    if (irglb == icglb)
    {
        value += diag;
    }
    return value;
}

// Local column-major block (leading dimension nrloc) of the test matrix.
std::vector<std::complex<double>> fill_local(const BlockCyclicDistribution& dist,
                                             const int seed,
                                             const double diag,
                                             const double scale)
{
    const std::size_t len = std::max<std::size_t>(dist.size(), 1);
    std::vector<std::complex<double>> local(len, std::complex<double>(0.0, 0.0));
    for (int jl = 0; jl < dist.ncloc(); ++jl)
    {
        for (int il = 0; il < dist.nrloc(); ++il)
        {
            const std::size_t idx = static_cast<std::size_t>(jl) * dist.nrloc() + il;
            local[idx] = hermitian_entry(seed, diag, scale, dist.irglb(il), dist.icglb(jl));
        }
    }
    return local;
}

double max_abs_diff_all_ranks(const std::vector<std::complex<double>>& a,
                              const std::vector<std::complex<double>>& b)
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

// Parameters: global order, block size, swap grid, propagator type.
class PropagatorDdlaTest : public testing::TestWithParam<std::tuple<int, int, bool, int>>
{
  protected:
    std::ofstream ofs_running;
};

TEST_P(PropagatorDdlaTest, MatchesScalapack)
{
    const int nlocal = std::get<0>(GetParam());
    const int nb = std::get<1>(GetParam());
    const bool swap_grid = std::get<2>(GetParam());
    const int propagator = std::get<3>(GetParam());

    Parallel_Orbitals pv;
    pv.para_2d().init(nlocal, nlocal, nb, MPI_COMM_WORLD, swap_grid);
    const BlockCyclicDistribution& dist = pv.para_2d();
    module_rt::DdlaContext ddla;
    ddla.bind(dist);

    // S is Hermitian positive definite (close to I); H is Hermitian.
    const std::vector<std::complex<double>> s_local = fill_local(dist, 11, 1.0, 0.1 / std::sqrt(nlocal));
    const std::vector<std::complex<double>> h_local = fill_local(dist, 12, 0.0, 1.0);
    const std::vector<std::complex<double>> h_last_local = fill_local(dist, 13, 0.0, 1.0);
    const std::size_t len = s_local.size();
    std::vector<std::complex<double>> u_ref(len, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> u_ddla(len, std::complex<double>(0.0, 0.0));

    // Short step so that the Taylor series stays accurate for |H| ~ sqrt(n).
    const double md_dt = 0.2 * ModuleBase::AU_to_FS;
    const int print_matrix = 0;
    module_rt::Propagator prop(propagator, &pv, md_dt);
    prop.compute_propagator(nlocal,
                            s_local.data(),
                            h_local.data(),
                            h_last_local.data(),
                            u_ref.data(),
                            ofs_running,
                            print_matrix);
    prop.compute_propagator_ddla(nlocal,
                                 s_local.data(),
                                 h_local.data(),
                                 h_last_local.data(),
                                 u_ddla.data(),
                                 ddla,
                                 ofs_running,
                                 print_matrix);

    double u_norm = 0.0;
    for (std::size_t i = 0; i < len; ++i)
    {
        u_norm = std::max(u_norm, std::abs(u_ref[i]));
    }
    Parallel_Reduce::reduce_max(u_norm);
    ASSERT_GT(u_norm, 0.1);

    const double diff = max_abs_diff_all_ranks(u_ref, u_ddla);
    EXPECT_LT(diff, 1.0e-12 * nlocal) << "propagator=" << propagator << " nlocal=" << nlocal << " nb=" << nb << " grid=" << dist.nrproc()
                                      << "x" << dist.ncproc();
}

INSTANTIATE_TEST_SUITE_P(Layouts,
                         PropagatorDdlaTest,
                         testing::Combine(testing::Values(4, 37, 64),
                                          testing::Values(4),
                                          testing::Values(false, true),
                                          testing::Values(0, 1, 2)));

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}

#endif // __DDLA
