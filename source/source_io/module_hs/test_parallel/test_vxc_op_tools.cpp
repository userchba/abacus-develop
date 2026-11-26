#include "source_io/module_hs/vxc_op_tools.h"
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

template <typename T> T conjugate(const T& value) { return value; }
template <> Complex conjugate(const Complex& value) { return std::conj(value); }
template <typename T> T coefficient(int row, int col)
{
    return T(0.02 * (row + 1) + 0.03 * (col + 1));
}
template <> Complex coefficient(int row, int col)
{
    return Complex(0.02 * (row + 1) + 0.03 * (col + 1), 0.01 * (row - 2 * col));
}
template <typename T> T potential(int row, int col)
{
    double diagonal = 0.0;
    if (row == col) { diagonal = 3.0; }
    return T(diagonal + 0.01 * (row + col));
}
template <> Complex potential(int row, int col)
{
    double diagonal = 0.0;
    if (row == col) { diagonal = 3.0; }
    return Complex(diagonal + 0.01 * (row + col), 0.02 * (row - col));
}

class VxcOpToolsDdlaTest : public testing::TestWithParam<std::tuple<int, bool, int>>
{
  protected:
    void SetUp() override
    {
        n = std::get<0>(GetParam());
        bands = std::max(1, n / 2);
        const bool swapped = std::get<1>(GetParam());
        const int block = std::min(n, 2);
        const int empty = pv.para_2d().init(n, n, block, MPI_COMM_WORLD, swapped);
        const bool local_empty = pv.para_2d().nrloc() == 0 || pv.para_2d().ncloc() == 0;
        EXPECT_EQ(empty, static_cast<int>(local_empty));
        const int leading = std::max(1, pv.para_2d().nrloc());
        pv.set_desc_wfc_Eij(n, bands, leading);
        BlockCyclicDistribution wave;
        wave.set(n, bands, block, pv.para_2d().ictxt());
        pv.nloc_wfc = wave.size();
        mo.set(bands, bands, block, pv.para_2d().ictxt());
#ifdef __DDLA
        const int backend = std::get<2>(GetParam());
        if (backend != 0)
        {
            const bool use_gpu = backend == 2;
            context.bind(pv.para_2d(), use_gpu);
            selected = &context;
        }
#endif
    }

    template <typename T> void projection()
    {
        const BlockCyclicDistribution& ao = pv.para_2d();
        const std::size_t ao_capacity = std::max<std::size_t>(ao.size(), 1);
        const std::size_t c_capacity = std::max<std::size_t>(pv.nloc_wfc, 1);
        std::vector<T> v(ao_capacity, T(0.0));
        std::vector<T> c(c_capacity, T(0.0));
        for (int col = 0; col < ao.ncloc(); ++col)
        {
            const int global_col = ao.icglb(col);
            for (int row = 0; row < ao.nrloc(); ++row)
            {
                const int global_row = ao.irglb(row);
                const int offset = col * ao.nrloc() + row;
                v[offset] = potential<T>(global_row, global_col);
                if (global_col < bands) { c[offset] = coefficient<T>(global_row, global_col); }
            }
        }
        const std::vector<T> before = c;
        const std::vector<T> result = ModuleIO::cVc(v.data(), c.data(), n, bands, pv, mo, selected);
        EXPECT_EQ(c, before);
        double error = 0.0;
        for (int col = 0; col < mo.ncloc(); ++col)
        {
            const int global_col = mo.icglb(col);
            for (int row = 0; row < mo.nrloc(); ++row)
            {
                const int global_row = mo.irglb(row);
                T expected(0.0);
                for (int i = 0; i < n; ++i)
                {
                    const T ci = coefficient<T>(i, global_row);
                    for (int j = 0; j < n; ++j)
                    {
                        const T cj = coefficient<T>(j, global_col);
                        expected += conjugate(ci) * potential<T>(i, j) * cj;
                    }
                }
                const int offset = col * mo.nrloc() + row;
                error = std::max(error, std::abs(result[offset] - expected));
            }
        }
        Parallel_Reduce::reduce_max(error);
        EXPECT_LT(error, 1.0e-11);
    }

    Parallel_Orbitals pv;
    BlockCyclicDistribution mo;
    const ModuleBase::DdlaContext* selected = nullptr;
    int n = 0;
    int bands = 0;
#ifdef __DDLA
    ModuleBase::DdlaContext context;
#endif
};

TEST_P(VxcOpToolsDdlaTest, RectangularReal) { projection<double>(); }
TEST_P(VxcOpToolsDdlaTest, RectangularComplex) { projection<Complex>(); }

#ifdef __DDLA
#if DDLA_HAS_GPU
const auto backends = testing::Values(0, 1, 2);
#else
const auto backends = testing::Values(0, 1);
#endif
#else
const auto backends = testing::Values(0);
#endif
INSTANTIATE_TEST_SUITE_P(Distributed, VxcOpToolsDdlaTest,
                         testing::Combine(testing::Values(2, 11), testing::Bool(), backends));
} // namespace
