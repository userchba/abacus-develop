#include "source_lcao/module_dftu/dftu_nao_pots.h"
#include "source_base/block_cyclic_distribution.h"
#include "source_base/module_external/ddla_linalg.h"
#include "source_base/parallel_reduce.h"
#ifdef __DDLA
#include "source_base/module_external/ddla_context.h"
#endif

#include <gtest/gtest.h>
#include <mpi.h>
#include <algorithm>
#include <complex>
#include <tuple>
#include <vector>

namespace
{
using Complex = std::complex<double>;
template <typename T> T onsite_entry(int row, int col)
{
    return T(0.03 * (row + 1) - 0.02 * col);
}
template <> Complex onsite_entry(int row, int col)
{
    return Complex(0.03 * (row + 1) - 0.02 * col, 0.05 * (row + col + 1));
}
template <typename T> T overlap_entry(int row, int col)
{
    double diagonal = 0.0;
    if (row == col) { diagonal = 2.0; }
    return T(diagonal + 0.04 * (row + col + 1));
}
template <> Complex overlap_entry(int row, int col)
{
    double diagonal = 0.0;
    if (row == col) { diagonal = 2.0; }
    return Complex(diagonal + 0.04 * (row + col + 1), 0.02 * (row - col));
}

class DftuPotentialDdlaTest : public testing::TestWithParam<std::tuple<int, bool, int>>
{
  protected:
    void SetUp() override
    {
        n = std::get<0>(GetParam());
        const bool swapped = std::get<1>(GetParam());
        const int block = std::min(n, 2);
        const int empty = dist.init(n, n, block, MPI_COMM_WORLD, swapped);
        const bool local_empty = dist.nrloc() == 0 || dist.ncloc() == 0;
        EXPECT_EQ(empty, static_cast<int>(local_empty));
#ifdef __DDLA
        const int backend = std::get<2>(GetParam());
        if (backend != 0)
        {
            const bool use_gpu = backend == 2;
            context.bind(dist, use_gpu);
            selected = &context;
        }
#endif
    }

    template <typename T> void product()
    {
        const std::size_t capacity = std::max<std::size_t>(dist.size(), 1);
        std::vector<T> onsite(capacity, T(0.0));
        std::vector<T> overlap(capacity, T(0.0));
        std::vector<T> output(capacity, T(7.0));
        for (int col = 0; col < dist.ncloc(); ++col)
        {
            const int global_col = dist.icglb(col);
            for (int row = 0; row < dist.nrloc(); ++row)
            {
                const int global_row = dist.irglb(row);
                const int offset = col * dist.nrloc() + row;
                onsite[offset] = onsite_entry<T>(global_row, global_col);
                overlap[offset] = overlap_entry<T>(global_row, global_col);
            }
        }
        const std::vector<T> overlap_before = overlap;
        DFTU_LCAO::cal_pot_uterm(onsite, overlap.data(), output.data(), dist, selected);
        EXPECT_EQ(overlap, overlap_before);
        double error = 0.0;
        for (int col = 0; col < dist.ncloc(); ++col)
        {
            const int global_col = dist.icglb(col);
            for (int row = 0; row < dist.nrloc(); ++row)
            {
                const int global_row = dist.irglb(row);
                T expected(0.0);
                for (int k = 0; k < n; ++k)
                {
                    expected += T(0.5) * onsite_entry<T>(global_row, k) * overlap_entry<T>(k, global_col);
                    expected += T(0.5) * onsite_entry<T>(global_col, k) * overlap_entry<T>(k, global_row);
                }
                const int offset = col * dist.nrloc() + row;
                error = std::max(error, std::abs(output[offset] - expected));
            }
        }
        if (dist.size() == 0) { EXPECT_EQ(output[0], T(7.0)); }
        Parallel_Reduce::reduce_max(error);
        EXPECT_LT(error, 1.0e-11);
    }

    void context_binding()
    {
        EXPECT_EQ(ModuleBase::ddla_linalg::bound_backend(nullptr, dist), nullptr);
        EXPECT_EQ(ModuleBase::ddla_linalg::bound_backend(selected, dist), selected);
#ifdef __DDLA
        ModuleBase::DdlaContext unbound;
        EXPECT_EQ(ModuleBase::ddla_linalg::bound_backend(&unbound, dist), nullptr);
#endif
    }

    BlockCyclicDistribution dist;
    const ModuleBase::DdlaContext* selected = nullptr;
    int n = 0;
#ifdef __DDLA
    ModuleBase::DdlaContext context;
#endif
};
TEST_P(DftuPotentialDdlaTest, Real) { product<double>(); }
TEST_P(DftuPotentialDdlaTest, NonconjugateComplexTranspose) { product<Complex>(); }
TEST_P(DftuPotentialDdlaTest, BoundContext) { context_binding(); }
#ifdef __DDLA
#if DDLA_HAS_GPU
const auto backends = testing::Values(0, 1, 2);
#else
const auto backends = testing::Values(0, 1);
#endif
#else
const auto backends = testing::Values(0);
#endif
INSTANTIATE_TEST_SUITE_P(Distributed, DftuPotentialDdlaTest,
                         testing::Combine(testing::Values(2, 11), testing::Bool(), backends));
} // namespace
