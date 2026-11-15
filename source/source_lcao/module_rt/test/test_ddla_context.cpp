#ifdef __DDLA

#include "../ddla_context.h"
#include "source_base/block_cyclic_distribution.h"
#include "source_base/module_external/scalapack_connector.h"
#include "source_base/parallel_reduce.h"

#include <ddla/ddla.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstddef>
#include <tuple>
#include <vector>

/************************************************
 *  unit test of module_rt::DdlaContext
 ***********************************************/

/**
 * - Tested functions
 *   - DdlaContext::bind / release / handle
 *     - binding twice to the same BLACS grid keeps one handle;
 *     - release unbinds and a later bind creates a new handle;
 *     - a GPU handle is bound when LibDDLA has GPU support.
 *   - LibDDLA pgemm on the bound handle
 *     - matches ScaLAPACK pzgemm on ABACUS block-cyclic layouts, for
 *       op(A) = A and A^H, uneven block counts, and swapped grids.
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

// Local column-major block (leading dimension nrloc) of the test matrix.
std::vector<std::complex<double>> fill_local(const BlockCyclicDistribution& dist, const int seed)
{
    const std::size_t len = std::max<std::size_t>(dist.size(), 1);
    std::vector<std::complex<double>> local(len, std::complex<double>(0.0, 0.0));
    for (int jl = 0; jl < dist.ncloc(); ++jl)
    {
        for (int il = 0; il < dist.nrloc(); ++il)
        {
            const std::size_t idx = static_cast<std::size_t>(jl) * dist.nrloc() + il;
            local[idx] = test_entry(seed, dist.irglb(il), dist.icglb(jl));
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

class DdlaContextTest : public testing::Test
{
};

TEST_F(DdlaContextTest, BindReusesHandleOnSameGrid)
{
    BlockCyclicDistribution dist;
    const int n = 20;
    const int nb = 4;
    const bool swap_grid = false;
    dist.init(n, n, nb, MPI_COMM_WORLD, swap_grid);

    module_rt::DdlaContext ctx;
    EXPECT_FALSE(ctx.is_bound());
    ctx.bind(dist, false);
    ASSERT_TRUE(ctx.is_bound());
    EXPECT_EQ(ctx.ictxt(), dist.ictxt());
    const ddla::DdlaHandle_t first = ctx.handle();

    ctx.bind(dist, false);
    EXPECT_EQ(ctx.handle(), first);

    ctx.release();
    EXPECT_FALSE(ctx.is_bound());
    EXPECT_EQ(ctx.ictxt(), -1);

    ctx.bind(dist, false);
    EXPECT_TRUE(ctx.is_bound());
    EXPECT_EQ(ddla::ddlaGetBackend(ctx.handle()), ddla::DdlaBackend::CPU);
}

#if DDLA_HAS_GPU
TEST_F(DdlaContextTest, BindsGpuHandle)
{
    BlockCyclicDistribution dist;
    const int n = 20;
    const int nb = 4;
    const bool swap_grid = false;
    dist.init(n, n, nb, MPI_COMM_WORLD, swap_grid);

    module_rt::DdlaContext ctx;
    ctx.bind(dist, true);
    ASSERT_TRUE(ctx.is_bound());
    EXPECT_TRUE(ctx.on_gpu());
    EXPECT_EQ(ddla::ddlaGetBackend(ctx.handle()), ddla::DdlaBackend::GPU);

    // Switching the backend on the same grid replaces the handle.
    ctx.bind(dist, false);
    EXPECT_FALSE(ctx.on_gpu());
    EXPECT_EQ(ddla::ddlaGetBackend(ctx.handle()), ddla::DdlaBackend::CPU);
}
#endif // DDLA_HAS_GPU

// Parameters: global order, block size, conjugate-transpose A, swap grid.
class DdlaContextPgemmTest : public testing::TestWithParam<std::tuple<int, int, bool, bool>>
{
};

TEST_P(DdlaContextPgemmTest, PgemmMatchesScalapack)
{
    const int n = std::get<0>(GetParam());
    const int nb = std::get<1>(GetParam());
    const bool conj_a = std::get<2>(GetParam());
    const bool swap_grid = std::get<3>(GetParam());

    BlockCyclicDistribution dist;
    dist.init(n, n, nb, MPI_COMM_WORLD, swap_grid);
    module_rt::DdlaContext ctx;
    ctx.bind(dist, false);

    const std::vector<std::complex<double>> a = fill_local(dist, 1);
    const std::vector<std::complex<double>> b = fill_local(dist, 2);
    std::vector<std::complex<double>> c_ref = fill_local(dist, 3);
    std::vector<std::complex<double>> c_ddla = c_ref;

    const char transa = conj_a ? 'C' : 'N';
    const char transb = 'N';
    const std::complex<double> alpha(0.75, -0.25);
    const std::complex<double> beta(0.5, 0.125);
    ScalapackConnector::gemm(transa,
                             transb,
                             n,
                             n,
                             n,
                             alpha,
                             a.data(),
                             1,
                             1,
                             dist.desc(),
                             b.data(),
                             1,
                             1,
                             dist.desc(),
                             beta,
                             c_ref.data(),
                             1,
                             1,
                             dist.desc());
    ddla::pgemm(ctx.handle(),
                transa,
                transb,
                n,
                n,
                n,
                alpha,
                a.data(),
                dist.desc(),
                b.data(),
                dist.desc(),
                beta,
                c_ddla.data(),
                dist.desc());

    const double diff = max_abs_diff_all_ranks(c_ref, c_ddla);
    EXPECT_LT(diff, 1.0e-12 * n) << "n=" << n << " nb=" << nb << " transa=" << transa
                                 << " grid=" << dist.nrproc() << "x" << dist.ncproc();
}

INSTANTIATE_TEST_SUITE_P(Layouts,
                         DdlaContextPgemmTest,
                         testing::Values(std::make_tuple(16, 4, false, false),
                                         std::make_tuple(37, 4, false, false),
                                         std::make_tuple(37, 4, true, false),
                                         std::make_tuple(64, 8, true, false),
                                         std::make_tuple(50, 16, false, true),
                                         std::make_tuple(50, 16, true, true)));

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}

#endif // __DDLA
