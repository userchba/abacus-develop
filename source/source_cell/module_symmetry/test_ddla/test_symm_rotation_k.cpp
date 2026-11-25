#include "source_cell/module_symmetry/symm_rotation_k.h"
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

Complex density_entry(int row, int col)
{
    return Complex(0.07 * (row + 1) + 0.03 * col, 0.02 * (row - col + 1));
}
Complex rotation_entry(int row, int col)
{
    double diagonal = 0.0;
    if (row == col) { diagonal = 1.0; }
    return Complex(diagonal + 0.01 * (row + col), 0.03 * (row - col + 1));
}

class SymmetryDdlaTest : public testing::TestWithParam<std::tuple<int, bool, int>>
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
        const std::size_t capacity = std::max<std::size_t>(dist.size(), 1);
        density.assign(capacity, Complex(0.0, 0.0));
        rotation.assign(capacity, Complex(0.0, 0.0));
        for (int col = 0; col < dist.ncloc(); ++col)
        {
            const int global_col = dist.icglb(col);
            for (int row = 0; row < dist.nrloc(); ++row)
            {
                const int global_row = dist.irglb(row);
                const int offset = col * dist.nrloc() + row;
                density[offset] = density_entry(global_row, global_col);
                rotation[offset] = rotation_entry(global_row, global_col);
            }
        }
        std::vector<std::map<int, std::vector<Complex>>> rotations(1);
        rotations[0][0] = rotation;
        std::vector<std::vector<int>> groups(1, std::vector<int>(1, 0));
        symmetry.set_density_rotations_for_testing(rotations, groups, 1, 4);
    }

    void spatial(bool antiunitary)
    {
        const std::vector<Complex> result = symmetry.rot_matrix_ao(density, 0, 3, 0, dist, antiunitary, selected);
        double error = 0.0;
        for (int col = 0; col < dist.ncloc(); ++col)
        {
            const int global_col = dist.icglb(col);
            for (int row = 0; row < dist.nrloc(); ++row)
            {
                const int global_row = dist.irglb(row);
                Complex expected(0.0, 0.0);
                for (int i = 0; i < n; ++i)
                {
                    for (int j = 0; j < n; ++j)
                    {
                        Complex value = density_entry(i, j);
                        if (antiunitary) { value = std::conj(value); }
                        expected += rotation_entry(i, global_row) * value * std::conj(rotation_entry(j, global_col));
                    }
                }
                expected /= 3.0;
                const int offset = col * dist.nrloc() + row;
                error = std::max(error, std::abs(result[offset] - expected));
            }
        }
        Parallel_Reduce::reduce_max(error);
        EXPECT_LT(error, 1.0e-11);
    }

    void spin_time_reversal()
    {
        const std::vector<Complex> spin = symmetry.set_sigma_y_2d(dist);
        const std::vector<Complex> result = symmetry.trs_spin_rotate(density, spin, dist, 0.5, selected);
        double error = 0.0;
        for (int col = 0; col < dist.ncloc(); ++col)
        {
            const int global_col = dist.icglb(col);
            const int flipped_col = global_col ^ 1;
            const double right_sign = 1 - 2 * (global_col % 2);
            for (int row = 0; row < dist.nrloc(); ++row)
            {
                const int global_row = dist.irglb(row);
                const int flipped_row = global_row ^ 1;
                const double left_sign = 2 * (global_row % 2) - 1;
                const Complex value = density_entry(flipped_row, flipped_col);
                const Complex expected = -0.5 * left_sign * right_sign * std::conj(value);
                const int offset = col * dist.nrloc() + row;
                error = std::max(error, std::abs(result[offset] - expected));
            }
        }
        Parallel_Reduce::reduce_max(error);
        EXPECT_LT(error, 1.0e-11);
    }

    BlockCyclicDistribution dist;
    ModuleSymmetry::Symmetry_rotation_k symmetry;
    std::vector<Complex> density;
    std::vector<Complex> rotation;
    const ModuleBase::DdlaContext* selected = nullptr;
    int n = 0;
#ifdef __DDLA
    ModuleBase::DdlaContext context;
#endif
};

TEST_P(SymmetryDdlaTest, Spatial) { spatial(false); }
TEST_P(SymmetryDdlaTest, Antiunitary) { spatial(true); }
TEST_P(SymmetryDdlaTest, SpinTimeReversal) { spin_time_reversal(); }
#ifdef __DDLA
#if DDLA_HAS_GPU
const auto backends = testing::Values(0, 1, 2);
#else
const auto backends = testing::Values(0, 1);
#endif
#else
const auto backends = testing::Values(0);
#endif
INSTANTIATE_TEST_SUITE_P(Distributed, SymmetryDdlaTest,
                         testing::Combine(testing::Values(2, 10), testing::Bool(), backends));
} // namespace
