#include "source_base/module_external/ddla_linalg.h"
#include "source_base/parallel_reduce.h"
#include "source_basis/module_ao/parallel_orbitals.h"
#ifdef __DDLA
#include "source_base/module_external/ddla_context.h"
#endif

#include <gtest/gtest.h>
#include <mpi.h>
#include <algorithm>
#include <complex>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace
{
using Complex = std::complex<double>;
using ModuleBase::ddla_linalg::ResidentMatrix;

template <typename T> T entry(int row, int col)
{
    return T(0.03 * (row + 1) - 0.02 * col);
}
template <> Complex entry<Complex>(int row, int col)
{
    return Complex(0.03 * (row + 1) - 0.02 * col, 0.01 * (row - col));
}
template <typename T>
std::vector<T> local_matrix(const BlockCyclicDistribution& dist)
{
    const std::size_t capacity = std::max<std::size_t>(dist.size(), 1);
    std::vector<T> data(capacity, T(0.0));
    for (int col = 0; col < dist.ncloc(); ++col)
    {
        for (int row = 0; row < dist.nrloc(); ++row)
        {
            const int global_row = dist.irglb(row);
            const int global_col = dist.icglb(col);
            data[col * dist.nrloc() + row] = entry<T>(global_row, global_col);
        }
    }
    return data;
}
template <typename T> double difference(const std::vector<T>& a, const std::vector<T>& b)
{
    double error = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        error = std::max(error, std::abs(a[i] - b[i]));
    }
    Parallel_Reduce::reduce_max(error);
    return error;
}

class DdlaResidentTest : public testing::TestWithParam<std::tuple<int, bool, int>>
{
  protected:
    void SetUp() override
    {
        n = std::get<0>(GetParam());
        bands = std::max(1, n / 2);
        const bool swapped = std::get<1>(GetParam());
        const int block = std::min(n, 2);
        pv.para_2d().init(n, n, block, MPI_COMM_WORLD, swapped);
        const int leading_dimension = std::max(1, pv.para_2d().nrloc());
        pv.set_desc_wfc_Eij(n, bands, leading_dimension);
#ifdef __DDLA
        const int backend = std::get<2>(GetParam());
        if (backend != 0)
        {
            const bool gpu = backend == 2;
            context.bind(pv.para_2d(), gpu);
            selected = &context;
            on_gpu = gpu;
        }
#endif
    }

    template <typename T> void chain()
    {
        const BlockCyclicDistribution& dist = pv.para_2d();
        const int* square = dist.desc();
        const int* wave = pv.desc_wfc;
        const int* band = pv.desc_Eij;
        std::vector<T> a = local_matrix<T>(dist);
        const std::vector<T> b = local_matrix<T>(dist);
        std::vector<T> reference(a.size(), T(7.0));
        std::vector<T> output = reference;
        std::vector<T> temporary(a.size(), T(0.0));
        const T one(1.0);
        const T zero(0.0);
        ResidentMatrix<T> resident_a(selected, a.data(), square);
        const ResidentMatrix<T> resident_b(selected, b.data(), wave);
        ResidentMatrix<T> intermediate(selected, wave);
        ResidentMatrix<T> result(selected, output.data(), band, false);
        T* initial_pointer = intermediate.data();
        const ResidentMatrix<T>& readonly_a = resident_a;
        if (on_gpu && dist.size() > 0)
        {
            EXPECT_NE(readonly_a.data(), a.data());
        }
        EXPECT_THROW(resident_a.data(), std::logic_error);
        for (int iteration = 0; iteration < 2; ++iteration)
        {
            ModuleBase::ddla_linalg::gemm(nullptr, 'N', 'N', n, bands, n, one, a.data(), square, b.data(), wave,
                                         zero, temporary.data(), wave);
            ModuleBase::ddla_linalg::gemm(nullptr, 'C', 'N', bands, bands, n, one, b.data(), wave, temporary.data(), wave,
                                         zero, reference.data(), band);
            ModuleBase::ddla_linalg::gemm('N', 'N', n, bands, n, one, resident_a, resident_b, zero, intermediate);
            ModuleBase::ddla_linalg::gemm('C', 'N', bands, bands, n, one, resident_b, intermediate, zero, result);
            if (on_gpu && iteration == 0)
            {
                const std::vector<T> untouched(output.size(), T(7.0));
                EXPECT_EQ(difference(output, untouched), 0.0);
            }
            result.copy_to_host(output.data());
            EXPECT_LT(difference(output, reference), 1.0e-12 * n);
            EXPECT_EQ(intermediate.data(), initial_pointer);
            for (T& value : a)
            {
                value *= 1.25;
            }
            resident_a.refresh(a.data());
        }
        ResidentMatrix<T> moved(std::move(intermediate));
        EXPECT_EQ(moved.data(), initial_pointer);
    }

    template <typename T> void offset()
    {
        const BlockCyclicDistribution& dist = pv.para_2d();
        const int* desc = dist.desc();
        const std::vector<T> a = local_matrix<T>(dist);
        std::vector<T> output = a;
        std::vector<T> reference = a;
        const T alpha(1.2);
        const T beta(0.4);
        const int length = n - 1;
        const ResidentMatrix<T> input(selected, a.data(), desc);
        ResidentMatrix<T> result(selected, output.data(), desc, true);
        ModuleBase::ddla_linalg::gemm(nullptr, 'N', 'N', length, 1, length, alpha, a.data(), 2, 2, desc,
                                     a.data(), 2, 1, desc, beta, reference.data(), 1, 2, desc);
        ModuleBase::ddla_linalg::gemm('N', 'N', length, 1, length, alpha, input, 2, 2, input, 2, 1,
                                     beta, result, 1, 2);
        result.copy_to_host(output.data());
        EXPECT_LT(difference(output, reference), 1.0e-12 * n);
    }

    template <typename T> void solve()
    {
        const BlockCyclicDistribution& dist = pv.para_2d();
        const int* desc = dist.desc();
        std::vector<T> s = local_matrix<T>(dist);
        for (int col = 0; col < dist.ncloc(); ++col)
        {
            for (int row = 0; row < dist.nrloc(); ++row)
            {
                if (dist.irglb(row) == dist.icglb(col))
                {
                    s[col * dist.nrloc() + row] += T(n + 1.0);
                }
            }
        }
        const std::vector<T> h = local_matrix<T>(dist);
        std::vector<T> reference = h;
        std::vector<T> s_reference = s;
        const int info = ModuleBase::ddla_linalg::gesv(nullptr, n, n, s_reference.data(), desc, reference.data(), desc);
        ASSERT_EQ(info, 0);
        std::vector<T> output = h;
        ResidentMatrix<T> factor(selected, s.data(), desc, true);
        ResidentMatrix<T> rhs(selected, output.data(), desc, true);
        if (selected == nullptr)
        {
            const int resident_info = ModuleBase::ddla_linalg::gesv(n, n, factor, rhs);
            ASSERT_EQ(resident_info, 0);
        }
        else
        {
            const ModuleBase::ddla_linalg::ResidentLU<T> lu(factor);
            ASSERT_EQ(lu.info(), 0);
            lu.solve('L', n, rhs);
            const ResidentMatrix<T> right_input(selected, h.data(), desc);
            ResidentMatrix<T> right(selected, desc);
            const T one(1.0);
            const T zero(0.0);
            ModuleBase::ddla_linalg::geadd('N', n, n, one, right_input, zero, right);
            lu.solve('R', n, right);
            std::vector<T> right_host(h.size(), T(0.0));
            right.copy_to_host(right_host.data());
            const std::vector<T> original_s = local_overlap<T>(dist);
            std::vector<T> residual(h.size(), T(0.0));
            ModuleBase::ddla_linalg::gemm(nullptr, 'N', 'N', n, n, n, one, right_host.data(), desc,
                                         original_s.data(), desc, zero, residual.data(), desc);
            EXPECT_LT(difference(residual, h), 1.0e-12 * n);
        }
        rhs.copy_to_host(output.data());
        EXPECT_LT(difference(output, reference), 1.0e-12 * n);
    }

    template <typename T> std::vector<T> local_overlap(const BlockCyclicDistribution& dist)
    {
        std::vector<T> s = local_matrix<T>(dist);
        for (int col = 0; col < dist.ncloc(); ++col)
        {
            for (int row = 0; row < dist.nrloc(); ++row)
            {
                if (dist.irglb(row) == dist.icglb(col))
                {
                    s[col * dist.nrloc() + row] += T(n + 1.0);
                }
            }
        }
        return s;
    }

    template <typename T> void determinant_pivots()
    {
        if (selected == nullptr) { GTEST_SKIP() << "LU pivot extraction selects an explicit DDLA backend"; }
        const BlockCyclicDistribution& dist = pv.para_2d();
        const std::size_t capacity = std::max<std::size_t>(dist.size(), 1);
        std::vector<T> factors(capacity, T(0.0));
        for (int col = 0; col < dist.ncloc(); ++col)
        {
            const int global_col = dist.icglb(col);
            int nonzero_row = global_col;
            if (global_col == 0) { nonzero_row = n - 1; }
            if (global_col == n - 1) { nonzero_row = 0; }
            const int next = global_col + 1;
            const T value = T(next) + entry<T>(global_col, next);
            for (int row = 0; row < dist.nrloc(); ++row)
            {
                if (dist.irglb(row) == nonzero_row)
                {
                    factors[col * dist.nrloc() + row] = value;
                }
            }
        }
        ResidentMatrix<T> matrix(selected, factors.data(), dist.desc(), true);
        ModuleBase::ddla_linalg::ResidentLU<T> lu(matrix);
        ASSERT_EQ(lu.info(), 0);
        std::vector<int> pivots;
        lu.copy_pivots_to_host(pivots);
        matrix.copy_to_host(factors.data());
        std::vector<T> diagonal(n, T(0.0));
        std::vector<double> global_pivots(n, 0.0);
        for (int i = 0; i < n; ++i)
        {
            if (dist.in_this_processor(i, i))
            {
                const int row = dist.irloc(i);
                const int col = dist.icloc(i);
                diagonal[i] = factors[col * dist.nrloc() + row];
                global_pivots[i] = pivots[row];
            }
        }
        Parallel_Reduce::reduce_all(diagonal.data(), n);
        Parallel_Reduce::reduce_all(global_pivots.data(), n);
        T determinant(1.0);
        T reference(-1.0);
        for (int i = 0; i < n; ++i)
        {
            const int next = i + 1;
            determinant *= diagonal[i];
            if (global_pivots[i] != next) { determinant = -determinant; }
            reference *= T(next) + entry<T>(i, next);
            EXPECT_GE(global_pivots[i], 1.0);
            EXPECT_LE(global_pivots[i], n);
        }
        const double tolerance = 1.0e-11 * std::max(1.0, std::abs(reference));
        EXPECT_LT(std::abs(determinant - reference), tolerance);
    }

    Parallel_Orbitals pv;
    const ModuleBase::DdlaContext* selected = nullptr;
    bool on_gpu = false;
    int n = 0;
    int bands = 0;
#ifdef __DDLA
    ModuleBase::DdlaContext context;
#endif
};

TEST_P(DdlaResidentTest, ChainAndRefreshReal) { chain<double>(); }
TEST_P(DdlaResidentTest, ChainAndRefreshComplex) { chain<Complex>(); }
TEST_P(DdlaResidentTest, OffsetReal) { offset<double>(); }
TEST_P(DdlaResidentTest, OffsetComplex) { offset<Complex>(); }
TEST_P(DdlaResidentTest, SolveReal) { solve<double>(); }
TEST_P(DdlaResidentTest, SolveComplex) { solve<Complex>(); }
TEST_P(DdlaResidentTest, DeterminantPivotsReal) { determinant_pivots<double>(); }
TEST_P(DdlaResidentTest, DeterminantPivotsComplex) { determinant_pivots<Complex>(); }

#ifdef __DDLA
#if DDLA_HAS_GPU
const auto backends = testing::Values(0, 1, 2);
#else
const auto backends = testing::Values(0, 1);
#endif
#else
const auto backends = testing::Values(0);
#endif
INSTANTIATE_TEST_SUITE_P(Distributed, DdlaResidentTest,
                        testing::Combine(testing::Values(2, 11), testing::Bool(), backends));
} // namespace
