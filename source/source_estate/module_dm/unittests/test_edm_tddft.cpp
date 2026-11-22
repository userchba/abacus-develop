#include "source_estate/module_dm/edm_tddft.h"
#include "source_base/module_external/ddla_context.h"
#include "source_base/parallel_reduce.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <utility>
#include <vector>

namespace
{
using Complex = std::complex<double>;

class MatrixHamiltonian : public hamilt::Hamilt<Complex>
{
  public:
    explicit MatrixHamiltonian(const BlockCyclicDistribution& distribution) : dist_(distribution)
    {
        h_.resize(dist_.size());
        s_.resize(dist_.size());
    }

    void updateHk(const int ik) override
    {
        for (int col = 0; col < dist_.ncloc(); ++col)
        {
            for (int row = 0; row < dist_.nrloc(); ++row)
            {
                const int i = dist_.irglb(row);
                const int j = dist_.icglb(col);
                const int index = col * dist_.nrloc() + row;
                h_[index] = Complex(0.1 * std::cos(i + j + ik), 0.01 * (i - j) * (ik + 1));
                s_[index] = Complex(0.001 * std::cos(i + j + ik), 0.0001 * (i - j));
                if (i == j)
                {
                    s_[index] += 2.0 + 0.1 * ik;
                }
            }
        }
    }

    void matrix(hamilt::MatrixBlock<Complex>& h, hamilt::MatrixBlock<Complex>& s) override
    {
        h.p = h_.data();
        h.row = dist_.nrloc();
        h.col = dist_.ncloc();
        h.desc = dist_.desc();
        s = h;
        s.p = s_.data();
    }

  private:
    const BlockCyclicDistribution& dist_;
    std::vector<Complex> h_;
    std::vector<Complex> s_;
};

class EdmTddftTest : public testing::TestWithParam<std::pair<bool, bool>>
{
};

TEST_P(EdmTddftTest, MatchesNativeAcrossKPoints)
{
    const bool use_gpu = GetParam().first;
    const bool swapped = GetParam().second;
    const int n = 13;
    const int nk = 3;
    Parallel_Orbitals orbitals;
    orbitals.para_2d().init(n, n, 2, MPI_COMM_WORLD, swapped);
    const BlockCyclicDistribution& dist = orbitals.para_2d();
    ModuleBase::DdlaContext context;
    context.bind(dist, use_gpu);
    K_Vectors kpoints;
    kpoints.set_nks(nk);
    kpoints.kvec_d.resize(nk);
    module_dm::Setup_DM<Complex> native;
    module_dm::Setup_DM<Complex> selected;
    native.dm = new module_dm::DensityMatrix<Complex, double>(&orbitals, 1, kpoints.kvec_d, nk, 1);
    selected.dm = new module_dm::DensityMatrix<Complex, double>(&orbitals, 1, kpoints.kvec_d, nk, 1);
    for (int k = 0; k < nk; ++k)
    {
        for (int col = 0; col < dist.ncloc(); ++col)
        {
            for (int row = 0; row < dist.nrloc(); ++row)
            {
                const int i = dist.irglb(row);
                const int j = dist.icglb(col);
                const int index = col * dist.nrloc() + row;
                const Complex value(0.03 * std::cos(i + j + k), 0.002 * (i - j));
                native.dm->get_dmk_ptr(k)[index] = value;
                selected.dm->get_dmk_ptr(k)[index] = value;
            }
        }
    }
    MatrixHamiltonian hamiltonian(dist);
    module_dm::edm_tddft(orbitals, nullptr, native, kpoints, &hamiltonian);
    module_dm::edm_tddft(orbitals, &context, selected, kpoints, &hamiltonian);
    double error = 0.0;
    double magnitude = 0.0;
    for (int k = 0; k < nk; ++k)
    {
        for (std::size_t i = 0; i < dist.size(); ++i)
        {
            const Complex reference = native.dm->edmk[k].c[i];
            const Complex actual = selected.dm->edmk[k].c[i];
            error = std::max(error, std::abs(actual - reference));
            magnitude = std::max(magnitude, std::abs(reference));
            EXPECT_EQ(selected.dm->get_dmk_ptr(k)[i], native.dm->get_dmk_ptr(k)[i]);
        }
    }
    Parallel_Reduce::reduce_max(error);
    Parallel_Reduce::reduce_max(magnitude);
    ASSERT_GT(magnitude, 1.0e-6);
    EXPECT_LT(error, 1.0e-12 * n);
}

#if DDLA_HAS_GPU
const auto modes = testing::Values(std::make_pair(false, false), std::make_pair(false, true),
                                  std::make_pair(true, false), std::make_pair(true, true));
#else
const auto modes = testing::Values(std::make_pair(false, false), std::make_pair(false, true));
#endif
INSTANTIATE_TEST_SUITE_P(Backends, EdmTddftTest, modes);
} // namespace
