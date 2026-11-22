#ifdef __DDLA

#include "source_base/module_external/ddla_context.h"
#include "source_base/module_external/ddla_linalg.h"
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
#include <type_traits>
#include <vector>

/************************************************
 *  unit test of ModuleBase::ddla_linalg
 ***********************************************/

/**
 * - Tested functions
 *   - gemm, geadd with a LibDDLA context
 *     - match the ScaLAPACK path (null context) for the products used by
 *       one Crank-Nicolson step: H*psi, psi^H*(S psi) into the band x band
 *       Eij layout, psi*Cij, and the H/S half-step average, on uneven
 *       blocks, nband < nlocal and swapped grids, on the CPU and (when
 *       available) the GPU backend.
 *   - getrf + getrs
 *     - side 'L' matches ScaLAPACK pzgesv for S X = B with B in the psi
 *       (nlocal x nband) layout;
 *     - side 'R' solves X S = H: the residual X S - H vanishes.
 *   - gemm on submatrices (ia/ja offsets): occupied/virtual column windows
 *     of psi as LR-TDDFT uses them, C with and without an offset and with
 *     beta zero and non-zero, real and complex.
 *   - tran 'T' and 'C' on square and rectangular matrices, real and complex.
 *   - gesv and posv match ScaLAPACK p?gesv and p?potrf + p?potrs, and posv
 *     reports a matrix that is not positive definite with the same info.
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


// Real matrices from the real parts of the complex test entries.
std::vector<double> real_part(const std::vector<Complex>& z)
{
    std::vector<double> x(z.size(), 0.0);
    for (std::size_t i = 0; i < z.size(); ++i)
    {
        x[i] = z[i].real();
    }
    return x;
}

double max_abs_diff_all_ranks(const std::vector<double>& a, const std::vector<double>& b)
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

// Run the backends LibDDLA was built with (false: CPU, true: GPU); the CPU
// backend is always built.
#if DDLA_HAS_GPU
const auto ddla_backends = testing::Values(false, true);
#else
const auto ddla_backends = testing::Values(false);
#endif

// Parameters: (nlocal, nband, block size, swap grid), GPU backend.
class DdlaLinalgTest : public testing::TestWithParam<std::tuple<std::tuple<int, int, int, bool>, bool>>
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
    ModuleBase::DdlaContext ddla;
};

TEST_P(DdlaLinalgTest, GemmMatchesScalapack)
{
    const BlockCyclicDistribution& dist = pv.para_2d();
    const std::vector<Complex> h = fill_local(dist, nlocal, nlocal, 1);
    const std::vector<Complex> psi = fill_local(dist, nlocal, nband, 2);
    const Complex one(1.0, 0.0);
    const Complex zero(0.0, 0.0);
    const ModuleBase::DdlaContext* scalapack = nullptr;

    // tmp = H * psi
    std::vector<Complex> tmp_ref(h.size(), zero);
    std::vector<Complex> tmp_ddla(h.size(), zero);
    ModuleBase::ddla_linalg::gemm(scalapack, 'N', 'N', nlocal, nband, nlocal, one, h.data(), dist.desc(), psi.data(),
                                  pv.desc_wfc, zero, tmp_ref.data(), pv.desc_wfc);
    ModuleBase::ddla_linalg::gemm(&ddla, 'N', 'N', nlocal, nband, nlocal, one, h.data(), dist.desc(), psi.data(),
                                  pv.desc_wfc, zero, tmp_ddla.data(), pv.desc_wfc);
    EXPECT_LT(max_abs_diff_all_ranks(tmp_ref, tmp_ddla), tol) << "H*psi";

    // eij = psi^H * tmp in the nband x nband Eij layout
    std::vector<Complex> eij_ref(h.size(), zero);
    std::vector<Complex> eij_ddla(h.size(), zero);
    ModuleBase::ddla_linalg::gemm(scalapack, 'C', 'N', nband, nband, nlocal, one, psi.data(), pv.desc_wfc, tmp_ref.data(),
                                  pv.desc_wfc, zero, eij_ref.data(), pv.desc_Eij);
    ModuleBase::ddla_linalg::gemm(&ddla, 'C', 'N', nband, nband, nlocal, one, psi.data(), pv.desc_wfc, tmp_ref.data(),
                                  pv.desc_wfc, zero, eij_ddla.data(), pv.desc_Eij);
    EXPECT_LT(max_abs_diff_all_ranks(eij_ref, eij_ddla), tol) << "psi^H*tmp";

    // psi_new = tmp * eij
    std::vector<Complex> out_ref(h.size(), zero);
    std::vector<Complex> out_ddla(h.size(), zero);
    ModuleBase::ddla_linalg::gemm(scalapack, 'N', 'N', nlocal, nband, nband, one, tmp_ref.data(), pv.desc_wfc,
                                  eij_ref.data(), pv.desc_Eij, zero, out_ref.data(), pv.desc_wfc);
    ModuleBase::ddla_linalg::gemm(&ddla, 'N', 'N', nlocal, nband, nband, one, tmp_ref.data(), pv.desc_wfc, eij_ref.data(),
                                  pv.desc_Eij, zero, out_ddla.data(), pv.desc_wfc);
    EXPECT_LT(max_abs_diff_all_ranks(out_ref, out_ddla), tol * nband) << "tmp*eij";
}

TEST_P(DdlaLinalgTest, GeaddMatchesScalapack)
{
    const BlockCyclicDistribution& dist = pv.para_2d();
    const std::vector<Complex> h_last = fill_local(dist, nlocal, nlocal, 3);
    std::vector<Complex> h_ref = fill_local(dist, nlocal, nlocal, 4);
    std::vector<Complex> h_ddla = h_ref;
    const Complex half(0.5, 0.0);
    const Complex coef(0.25, -0.75);
    const ModuleBase::DdlaContext* scalapack = nullptr;

    ModuleBase::ddla_linalg::geadd(scalapack, 'N', nlocal, nlocal, half, h_last.data(), dist.desc(), coef, h_ref.data(),
                                   dist.desc());
    ModuleBase::ddla_linalg::geadd(&ddla, 'N', nlocal, nlocal, half, h_last.data(), dist.desc(), coef, h_ddla.data(),
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

TEST_P(DdlaLinalgTest, GetrsMatchesScalapack)
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
    ModuleBase::ddla_linalg::getrf(ddla, dist, s_lu.data(), ipiv);
    ModuleBase::ddla_linalg::getrs(ddla, dist, left, nband, s_lu.data(), ipiv, x_ddla.data(), pv.desc_wfc);
    EXPECT_LT(max_abs_diff_all_ranks(x_ref, x_ddla), tol) << "S X = B";

    // Right solve X S = H with the same factors; check the residual X S - H.
    std::vector<Complex> x_right = h;
    ModuleBase::ddla_linalg::getrs(ddla, dist, right, nlocal, s_lu.data(), ipiv, x_right.data(), dist.desc());
    std::vector<Complex> residual = h;
    const ModuleBase::DdlaContext* scalapack = nullptr;
    ModuleBase::ddla_linalg::gemm(scalapack, 'N', 'N', nlocal, nlocal, nlocal, one, x_right.data(), dist.desc(), s.data(),
                                  dist.desc(), minus_one, residual.data(), dist.desc());
    const std::vector<Complex> zeros(residual.size(), Complex(0.0, 0.0));
    EXPECT_LT(max_abs_diff_all_ranks(residual, zeros), tol) << "X S = H";
}

TEST_P(DdlaLinalgTest, RealGemmAndGeaddMatchScalapack)
{
    const BlockCyclicDistribution& dist = pv.para_2d();
    const ModuleBase::DdlaContext* scalapack = nullptr;
    const std::vector<double> h = real_part(fill_local(dist, nlocal, nlocal, 1));
    const std::vector<double> psi = real_part(fill_local(dist, nlocal, nband, 2));
    const double one = 1.0;
    const double zero = 0.0;

    // beta = 0: the old content of C (7 here) must not leak into the result,
    // and the padding rows and columns outside the matrix keep it.
    std::vector<double> hpsi_ref(psi.size(), 7.0);
    std::vector<double> hpsi_ddla(psi.size(), 7.0);
    ModuleBase::ddla_linalg::gemm(scalapack, 'N', 'N', nlocal, nband, nlocal, one, h.data(), dist.desc(), psi.data(),
                                  pv.desc_wfc, zero, hpsi_ref.data(), pv.desc_wfc);
    ModuleBase::ddla_linalg::gemm(&ddla, 'N', 'N', nlocal, nband, nlocal, one, h.data(), dist.desc(), psi.data(),
                                  pv.desc_wfc, zero, hpsi_ddla.data(), pv.desc_wfc);
    EXPECT_LT(max_abs_diff_all_ranks(hpsi_ddla, hpsi_ref), tol);

    // C = 0.5 C + 2 H^T
    std::vector<double> c_ref = real_part(fill_local(dist, nlocal, nlocal, 3));
    std::vector<double> c_ddla = c_ref;
    const double two = 2.0;
    const double half = 0.5;
    ModuleBase::ddla_linalg::geadd(scalapack, 'T', nlocal, nlocal, two, h.data(), dist.desc(), half, c_ref.data(),
                                   dist.desc());
    ModuleBase::ddla_linalg::geadd(&ddla, 'T', nlocal, nlocal, two, h.data(), dist.desc(), half, c_ddla.data(),
                                   dist.desc());
    EXPECT_LT(max_abs_diff_all_ranks(c_ddla, c_ref), tol);
}

namespace
{
// psi-shaped (nlocal x nband) layout with the blocks and the grid of dist.
BlockCyclicDistribution layout(const BlockCyclicDistribution& dist, const int nrow, const int ncol)
{
    BlockCyclicDistribution out;
    out.set(nrow, ncol, dist.nb(), dist.ictxt());
    return out;
}

template <typename T>
std::vector<T> as_type(const std::vector<Complex>& z);
template <>
std::vector<Complex> as_type<Complex>(const std::vector<Complex>& z)
{
    return z;
}
template <>
std::vector<double> as_type<double>(const std::vector<Complex>& z)
{
    return real_part(z);
}

// Products on column windows of psi (occupied 1..nocc, virtual nocc+1..nband)
// with A, B and C offsets, LibDDLA against ScaLAPACK.
template <typename T>
void check_submatrix_gemm(const ModuleBase::DdlaContext& ddla, const Parallel_Orbitals& pv, const int nlocal,
                          const int nband, const double tol)
{
    const BlockCyclicDistribution& dist = pv.para_2d();
    const BlockCyclicDistribution pw = layout(dist, nlocal, nband);
    const BlockCyclicDistribution pb = layout(dist, nband, nband);
    const int nocc = std::max(1, nband / 3);
    const int nvirt = nband - nocc;
    const char adj = std::is_same<T, double>::value ? 'T' : 'C';
    const ModuleBase::DdlaContext* scalapack = nullptr;
    const std::vector<T> h = as_type<T>(fill_local(dist, nlocal, nlocal, 11));
    const std::vector<T> psi = as_type<T>(fill_local(pw, nlocal, nband, 12));
    const std::vector<T> phi = as_type<T>(fill_local(pw, nlocal, nband, 13));
    const T one(1.0);
    const T zero(0.0);
    const T half(0.5);

    // C(nlocal x nlocal) = psi_virt * phi_occ^H, k = nvirt (A offset)
    {
        std::vector<T> ref(std::max<std::size_t>(dist.size(), 1), T(3.0));
        std::vector<T> out = ref;
        ModuleBase::ddla_linalg::gemm(scalapack, 'N', adj, nlocal, nlocal, std::min(nocc, nvirt), one, psi.data(), 1,
                                      nocc + 1, pw.desc(), phi.data(), 1, 1, pw.desc(), zero, ref.data(), 1, 1,
                                      dist.desc());
        ModuleBase::ddla_linalg::gemm(&ddla, 'N', adj, nlocal, nlocal, std::min(nocc, nvirt), one, psi.data(), 1,
                                      nocc + 1, pw.desc(), phi.data(), 1, 1, pw.desc(), zero, out.data(), 1, 1,
                                      dist.desc());
        EXPECT_LT(max_abs_diff_all_ranks(ref, out), tol) << "psi_virt phi_occ^H";
    }
    // C(nvirt x nocc, leading block of the band layout) = psi_virt^H H phi_occ-like: A and B offsets, beta != 0
    {
        std::vector<T> hphi(std::max<std::size_t>(pw.size(), 1), zero);
        ModuleBase::ddla_linalg::gemm(scalapack, 'N', 'N', nlocal, nband, nlocal, one, h.data(), dist.desc(),
                                      phi.data(), pw.desc(), zero, hphi.data(), pw.desc());
        std::vector<T> ref = as_type<T>(fill_local(pb, nband, nband, 14));
        std::vector<T> out = ref;
        ModuleBase::ddla_linalg::gemm(scalapack, adj, 'N', nvirt, nocc, nlocal, half, psi.data(), 1, nocc + 1,
                                      pw.desc(), hphi.data(), 1, 1, pw.desc(), half, ref.data(), 1, 1, pb.desc());
        ModuleBase::ddla_linalg::gemm(&ddla, adj, 'N', nvirt, nocc, nlocal, half, psi.data(), 1, nocc + 1, pw.desc(),
                                      hphi.data(), 1, 1, pw.desc(), half, out.data(), 1, 1, pb.desc());
        EXPECT_LT(max_abs_diff_all_ranks(ref, out), tol) << "psi_virt^H H phi_occ";
    }
    // C(:, nocc+1:) = H * psi(:, 2:nvirt+1) + beta C(:, nocc+1:): B and C offsets, beta 0 and != 0
    for (const bool keep : {false, true})
    {
        const T beta = keep ? half : zero;
        std::vector<T> ref = as_type<T>(fill_local(pw, nlocal, nband, 15));
        std::vector<T> out = ref;
        ModuleBase::ddla_linalg::gemm(scalapack, 'N', 'N', nlocal, nvirt, nlocal, one, h.data(), 1, 1, dist.desc(),
                                      psi.data(), 1, 2, pw.desc(), beta, ref.data(), 1, nocc + 1, pw.desc());
        ModuleBase::ddla_linalg::gemm(&ddla, 'N', 'N', nlocal, nvirt, nlocal, one, h.data(), 1, 1, dist.desc(),
                                      psi.data(), 1, 2, pw.desc(), beta, out.data(), 1, nocc + 1, pw.desc());
        EXPECT_LT(max_abs_diff_all_ranks(ref, out), tol) << "C offset, beta " << (keep ? "0.5" : "0");
    }
}
} // namespace

TEST_P(DdlaLinalgTest, SubmatrixGemmMatchesScalapack)
{
    check_submatrix_gemm<double>(ddla, pv, nlocal, nband, tol);
    check_submatrix_gemm<Complex>(ddla, pv, nlocal, nband, tol);
}

TEST_P(DdlaLinalgTest, TranMatchesScalapack)
{
    const BlockCyclicDistribution& dist = pv.para_2d();
    const BlockCyclicDistribution pw = layout(dist, nlocal, nband);
    const BlockCyclicDistribution pt = layout(dist, nband, nlocal);
    const ModuleBase::DdlaContext* scalapack = nullptr;
    const std::vector<Complex> h = fill_local(dist, nlocal, nlocal, 21);
    const std::vector<Complex> psi = fill_local(pw, nlocal, nband, 22);
    const Complex alpha(0.5, 0.25);
    const Complex beta(-0.5, 0.0);
    for (const char trans : {'T', 'C'})
    {
        // square, accumulating into C (the matsym pattern)
        std::vector<Complex> ref = fill_local(dist, nlocal, nlocal, 23);
        std::vector<Complex> out = ref;
        ModuleBase::ddla_linalg::tran(scalapack, trans, nlocal, nlocal, alpha, h.data(), dist.desc(), beta, ref.data(),
                                      dist.desc());
        ModuleBase::ddla_linalg::tran(&ddla, trans, nlocal, nlocal, alpha, h.data(), dist.desc(), beta, out.data(),
                                      dist.desc());
        EXPECT_LT(max_abs_diff_all_ranks(ref, out), 1.0e-15) << "square " << trans;
        // rectangular nband x nlocal = op(psi)
        std::vector<Complex> tref(std::max<std::size_t>(pt.size(), 1), Complex(0.0, 0.0));
        std::vector<Complex> tout = tref;
        ModuleBase::ddla_linalg::tran(scalapack, trans, nband, nlocal, alpha, psi.data(), pw.desc(), Complex(0.0, 0.0),
                                      tref.data(), pt.desc());
        ModuleBase::ddla_linalg::tran(&ddla, trans, nband, nlocal, alpha, psi.data(), pw.desc(), Complex(0.0, 0.0),
                                      tout.data(), pt.desc());
        EXPECT_LT(max_abs_diff_all_ranks(tref, tout), 1.0e-15) << "rectangular " << trans;
        // real: 'C' is the plain transpose (PxTRAN)
        const std::vector<double> hr = real_part(h);
        std::vector<double> rref = real_part(fill_local(dist, nlocal, nlocal, 24));
        std::vector<double> rout = rref;
        ModuleBase::ddla_linalg::tran(scalapack, trans, nlocal, nlocal, 0.5, hr.data(), dist.desc(), 0.5, rref.data(),
                                      dist.desc());
        ModuleBase::ddla_linalg::tran(&ddla, trans, nlocal, nlocal, 0.5, hr.data(), dist.desc(), 0.5, rout.data(),
                                      dist.desc());
        EXPECT_LT(max_abs_diff_all_ranks(rref, rout), 1.0e-15) << "real " << trans;
    }
}

namespace
{
template <typename T>
void check_solvers(const ModuleBase::DdlaContext& ddla, const Parallel_Orbitals& pv, const int nlocal, const int nband,
                   const std::vector<Complex>& s_complex, const double tol)
{
    const BlockCyclicDistribution& dist = pv.para_2d();
    const BlockCyclicDistribution pw = layout(dist, nlocal, nband);
    const ModuleBase::DdlaContext* scalapack = nullptr;
    const std::vector<T> s = as_type<T>(s_complex);
    const std::vector<T> b = as_type<T>(fill_local(pw, nlocal, nband, 31));

    std::vector<T> a_ref = s;
    std::vector<T> x_ref = b;
    std::vector<T> a_out = s;
    std::vector<T> x_out = b;
    EXPECT_EQ(ModuleBase::ddla_linalg::gesv(scalapack, nlocal, nband, a_ref.data(), dist.desc(), x_ref.data(), pw.desc()), 0);
    EXPECT_EQ(ModuleBase::ddla_linalg::gesv(&ddla, nlocal, nband, a_out.data(), dist.desc(), x_out.data(), pw.desc()), 0);
    EXPECT_LT(max_abs_diff_all_ranks(x_ref, x_out), tol) << "gesv";

    a_ref = s;
    x_ref = b;
    a_out = s;
    x_out = b;
    EXPECT_EQ(ModuleBase::ddla_linalg::posv(scalapack, 'U', nlocal, nband, a_ref.data(), dist.desc(), x_ref.data(), pw.desc()), 0);
    EXPECT_EQ(ModuleBase::ddla_linalg::posv(&ddla, 'U', nlocal, nband, a_out.data(), dist.desc(), x_out.data(), pw.desc()), 0);
    EXPECT_LT(max_abs_diff_all_ranks(x_ref, x_out), tol) << "posv";

    // -S is negative definite: both report the first leading minor.
    std::vector<T> neg_ref = s;
    for (T& v : neg_ref)
    {
        v = -v;
    }
    std::vector<T> neg_out = neg_ref;
    x_ref = b;
    x_out = b;
    const int info_ref = ModuleBase::ddla_linalg::posv(scalapack, 'U', nlocal, nband, neg_ref.data(), dist.desc(), x_ref.data(), pw.desc());
    const int info_out = ModuleBase::ddla_linalg::posv(&ddla, 'U', nlocal, nband, neg_out.data(), dist.desc(), x_out.data(), pw.desc());
    EXPECT_GT(info_ref, 0);
    EXPECT_EQ(info_out, info_ref);
}
} // namespace

TEST_P(DdlaLinalgTest, GesvAndPosvMatchScalapack)
{
    const std::vector<Complex> s = overlap_local(pv.para_2d(), nlocal, 32);
    check_solvers<double>(ddla, pv, nlocal, nband, s, tol);
    check_solvers<Complex>(ddla, pv, nlocal, nband, s, tol);
}

INSTANTIATE_TEST_SUITE_P(Layouts,
                         DdlaLinalgTest,
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
