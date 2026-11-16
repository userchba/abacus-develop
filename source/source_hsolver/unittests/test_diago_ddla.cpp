#ifdef __DDLA

#include "source_base/block_cyclic_distribution.h"
#include "source_base/matrix_block.h"
#include "source_base/module_external/ddla_context.h"
#include "source_base/parallel_reduce.h"
#include "source_hsolver/diago_ddla.h"
#include "source_hsolver/unittests/diago_elpa_utils.h"
#include "source_psi/psi.h"

#include <ddla/ddla.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

/************************************************
 *  unit test of hsolver::DiagoDdla
 ***********************************************/

/**
 * - Tested function
 *   - DiagoDdla::diag on the Si2 and Si64 H/S matrices, gamma-only (double)
 *     and k-point (complex), for the CPU and/or GPU LibDDLA backend:
 *     - the lowest nbands eigenvalues match LAPACK hegv;
 *     - the eigenvectors satisfy H C = S C diag(e) and C^H S C = I;
 *     - the caller's H and S are left unchanged.
 */

namespace
{

double conj_value(const double x)
{
    return x;
}

std::complex<double> conj_value(const std::complex<double>& x)
{
    return std::conj(x);
}

struct DiagCase
{
    std::string hfile;
    std::string sfile;
    int nb;
};

// Reads a global H or S matrix (element (i, j) at i * n + j) and completes its
// lower triangle as the conjugate of the upper one.
template <typename T>
std::vector<T> read_hermitian(const std::string& fname, int& n)
{
    std::vector<T> matrix;
    const bool ok = LCAO_DIAGO_TEST::read_hs<std::vector<T>>(fname, matrix);
    EXPECT_TRUE(ok) << fname;
    n = static_cast<int>(std::lround(std::sqrt(static_cast<double>(matrix.size()))));
    for (int i = 0; i < n; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            matrix[j * n + i] = conj_value(matrix[i * n + j]);
        }
    }
    return matrix;
}

// Local column-major block (leading dimension lld) of a global matrix.
template <typename T>
std::vector<T> local_block(const BlockCyclicDistribution& dist, const std::vector<T>& global, const int n)
{
    const int lld = dist.desc()[8];
    std::vector<T> local(static_cast<std::size_t>(lld) * std::max(dist.ncloc(), 1), T(0.0));
    for (int jl = 0; jl < dist.ncloc(); ++jl)
    {
        for (int il = 0; il < dist.nrloc(); ++il)
        {
            local[static_cast<std::size_t>(jl) * lld + il] = global[dist.irglb(il) * n + dist.icglb(jl)];
        }
    }
    return local;
}

template <typename T>
void run_case(const DiagCase& dc, const bool gpu)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int n = 0;
    std::vector<T> h = read_hermitian<T>(dc.hfile, n);
    std::vector<T> s = read_hermitian<T>(dc.sfile, n);
    ASSERT_GT(n, 0);
    const int nbands = n / 2;

    BlockCyclicDistribution dist;
    dist.init(n, n, dc.nb, MPI_COMM_WORLD, false);
    const int lld = dist.desc()[8];
    const std::vector<T> h_local = local_block(dist, h, n);
    const std::vector<T> s_local = local_block(dist, s, n);
    std::vector<T> h_work = h_local;
    std::vector<T> s_work = s_local;
    ModuleBase::MatrixBlock<T> hk{h_work.data(),
                                  static_cast<std::size_t>(dist.nrloc()),
                                  static_cast<std::size_t>(dist.ncloc()),
                                  dist.desc()};
    ModuleBase::MatrixBlock<T> sk{s_work.data(),
                                  static_cast<std::size_t>(dist.nrloc()),
                                  static_cast<std::size_t>(dist.ncloc()),
                                  dist.desc()};

    int ncol_bands = 0;
    for (int jl = 0; jl < dist.ncloc(); ++jl)
    {
        if (dist.icglb(jl) < nbands)
        {
            ++ncol_bands;
        }
    }
    psi::Psi<T> psi;
    psi.resize(1, std::max(ncol_bands, 1), lld);

    ModuleBase::DdlaContext ddla;
    ddla.bind(dist, gpu);
    std::vector<double> e(nbands, 0.0);
    hsolver::DiagoDdla<T> solver(ddla, n, nbands);
    solver.diag(hk, sk, psi, e.data());

    EXPECT_EQ(h_work, h_local);
    EXPECT_EQ(s_work, s_local);

    // Global eigenvectors C(i, j) at i * nbands + j on every rank.
    std::vector<T> c(static_cast<std::size_t>(n) * nbands, T(0.0));
    for (int jl = 0; jl < ncol_bands; ++jl)
    {
        for (int il = 0; il < dist.nrloc(); ++il)
        {
            c[dist.irglb(il) * nbands + dist.icglb(jl)] = psi.get_pointer()[static_cast<std::size_t>(jl) * lld + il];
        }
    }
    const int c_size = static_cast<int>(c.size());
    Parallel_Reduce::reduce_all(c.data(), c_size);

    if (rank != 0)
    {
        return;
    }
    std::vector<double> e_ref(n, 0.0);
    std::vector<T> h_ref = h;
    std::vector<T> s_ref = s;
    LCAO_DIAGO_TEST::lapack_diago(h_ref.data(), s_ref.data(), e_ref.data(), n);
    double eig_err = 0.0;
    for (int i = 0; i < nbands; ++i)
    {
        eig_err = std::max(eig_err, std::abs(e[i] - e_ref[i]));
    }
    EXPECT_LT(eig_err, 1e-9);

    // hc = H C, sc = S C; residual H C - S C diag(e) and overlap C^H S C - I.
    double h_norm = 0.0;
    for (const T& x: h)
    {
        h_norm = std::max(h_norm, std::abs(x));
    }
    std::vector<T> sc(static_cast<std::size_t>(n) * nbands, T(0.0));
    double residual = 0.0;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < nbands; ++j)
        {
            T hc = T(0.0);
            T sc_ij = T(0.0);
            for (int k = 0; k < n; ++k)
            {
                hc += h[i * n + k] * c[k * nbands + j];
                sc_ij += s[i * n + k] * c[k * nbands + j];
            }
            sc[i * nbands + j] = sc_ij;
            residual = std::max(residual, std::abs(hc - sc_ij * e[j]));
        }
    }
    double overlap = 0.0;
    for (int a = 0; a < nbands; ++a)
    {
        for (int b = 0; b < nbands; ++b)
        {
            T csc = T(0.0);
            for (int k = 0; k < n; ++k)
            {
                csc += conj_value(c[k * nbands + a]) * sc[k * nbands + b];
            }
            const double target = (a == b) ? 1.0 : 0.0;
            overlap = std::max(overlap, std::abs(csc - T(target)));
        }
    }
    EXPECT_LT(residual / h_norm, 1e-8);
    EXPECT_LT(overlap, 1e-9);
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

const std::vector<DiagCase> gamma_cases = {{"H-GammaOnly-Si2.dat", "S-GammaOnly-Si2.dat", 1},
                                           {"H-GammaOnly-Si64.dat", "S-GammaOnly-Si64.dat", 32},
                                           {"H-GammaOnly-Si64.dat", "S-GammaOnly-Si64.dat", 7}};
const std::vector<DiagCase> kpoint_cases = {{"H-KPoints-Si2.dat", "S-KPoints-Si2.dat", 1},
                                            {"H-KPoints-Si2.dat", "S-KPoints-Si2.dat", 3},
                                            {"H-KPoints-Si64.dat", "S-KPoints-Si64.dat", 32}};

} // namespace

class DiagoDdlaTest : public ::testing::TestWithParam<std::tuple<int, bool>>
{
};

TEST_P(DiagoDdlaTest, GammaOnly)
{
    run_case<double>(gamma_cases[std::get<0>(GetParam())], std::get<1>(GetParam()));
}

TEST_P(DiagoDdlaTest, KPoints)
{
    run_case<std::complex<double>>(kpoint_cases[std::get<0>(GetParam())], std::get<1>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(Cases,
                         DiagoDdlaTest,
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
