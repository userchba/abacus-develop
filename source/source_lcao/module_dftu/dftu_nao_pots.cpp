#include "source_base/module_external/ddla_linalg.h"
#include "source_pw/module_pwdft/dftu_base.h"
#include "dftu_nao_pots.h"

#include "source_base/global_function.h"
#include "source_base/module_external/scalapack_connector.h"
#include "source_base/timer.h"
#include "source_base/tool_title.h"
#include "source_basis/module_ao/parallel_orbitals.h"
#include "source_cell/unitcell.h"

#include <algorithm>

/// On-site potential and Hubbard energy for one correlated shell:
///   pot_onsite(m,m') = U_eff * (0.5 * delta_{m,m'} - occ(m,m'))
///   EU = (U_eff / 2) * sum_{m,m'} occ(m,m') * (delta_{m,m'} - occ(m',m))
void DFTU_LCAO::cal_pot_onsite(const std::vector<double>& occ, const int m_size, const double u_value,
                               double* pot_onsite, double& eu)
{
    int spin_fold = occ.size() / m_size / m_size;
    if (spin_fold < 4) {
        for (int is = 0; is < spin_fold; ++is)
        {
            int start = is * m_size * m_size;
            for (int m1 = 0; m1 < m_size; m1++)
            {
                for (int m2 = 0; m2 < m_size; m2++)
                {
                    pot_onsite[start + m1 * m_size + m2] = u_value * (0.5 * (m1 == m2) - occ[start + m2 * m_size + m1]);
                    eu += u_value * 0.5 * occ[start + m2 * m_size + m1] * occ[start + m1 * m_size + m2];
                }
            }
        }
    } else
    {
        for (int m1 = 0; m1 < m_size; m1++)
        {
            for (int m2 = 0; m2 < m_size; m2++)
            {
                pot_onsite[m1 * m_size + m2] = u_value * (1.0 * (m1 == m2) - occ[m2 * m_size + m1]);
                eu += u_value * 0.25 * occ[m2 * m_size + m1] * occ[m1 * m_size + m2];
            }
        }
        for (int is = 1; is < spin_fold; ++is)
        {
            int start = is * m_size * m_size;
            for (int m1 = 0; m1 < m_size; m1++)
            {
                for (int m2 = 0; m2 < m_size; m2++)
                {
                    pot_onsite[start + m1 * m_size + m2] = u_value * (0 - occ[start + m2 * m_size + m1]);
                    eu += u_value * 0.25 * occ[start + m2 * m_size + m1] * occ[start + m1 * m_size + m2];
                }
            }
        }
    }
}

template <typename T>
void DFTU_LCAO::cal_pot_onsite(const Plus_U_Base& dftu,
                           const UnitCell& ucell,
                           const Parallel_Orbitals* pv,
                           const int spin,
                           const bool new_occ_mat,
                           T* pot_onsite)
{
    ModuleBase::TITLE("DFTU_LCAO", "pot_onsite");
    std::fill(pot_onsite, pot_onsite + pv->para_2d().size(), T(0));

    const int npol = dftu.occmat().npol();

    for (int it = 0; it < ucell.ntype; ++it)
    {
        if (dftu.get_l_channel(it) == -1)
        {
            continue;
        }
        for (int ia = 0; ia < ucell.atoms[it].na; ia++)
        {
            const int iat = ucell.itia2iat(it, ia);
            for (int L = 0; L <= ucell.atoms[it].nwl; L++)
            {
                if (L != dftu.get_l_channel(it))
                {
                    continue;
                }

                for (int m1 = 0; m1 < 2 * L + 1; m1++)
                {
                    for (int ipol1 = 0; ipol1 < npol; ipol1++)
                    {
                        const int mu = pv->para_2d().irloc(dftu.occmat().corr_iwt(iat, L, m1, ipol1));
                        if (mu < 0)
                        {
                            continue;
                        }

                        for (int m2 = 0; m2 < 2 * L + 1; m2++)
                        {
                            for (int ipol2 = 0; ipol2 < npol; ipol2++)
                            {
                                const int nu
                                    = pv->para_2d().icloc(dftu.occmat().corr_iwt(iat, L, m2, ipol2));
                                if (nu < 0)
                                {
                                    continue;
                                }
                                int m1_all = m1 + (2 * L + 1) * ipol1;
                                int m2_all = m2 + (2 * L + 1) * ipol2;
                                double val = get_onsite_pot(dftu, it, iat, L, spin,
                                                            m1_all, m2_all, new_occ_mat);
                                pot_onsite[nu * pv->para_2d().nrloc() + mu] = static_cast<T>(val);
                            } // ipol2
                        } // m2
                    } // ipol1
                } // m1
            } // l
        } // ia
    } // it

    return;
}

// Explicit instantiation
template void DFTU_LCAO::cal_pot_onsite<double>(const Plus_U_Base& dftu,
                                                const UnitCell& ucell,
                                                const Parallel_Orbitals* pv,
                                                const int spin,
                                                const bool new_occ_mat,
                                                double* pot_onsite);
template void DFTU_LCAO::cal_pot_onsite<std::complex<double>>(const Plus_U_Base& dftu,
                                                              const UnitCell& ucell,
                                                              const Parallel_Orbitals* pv,
                                                              const int spin,
                                                              const bool new_occ_mat,
                                                              std::complex<double>* pot_onsite);

template <typename T>
void DFTU_LCAO::cal_pot_uterm(Plus_U_Base& dftu,
                              const UnitCell& ucell,
                              const Parallel_Orbitals* pv,
                              const int spin,
                              T* pot_uterm,
                              const T* sk)
{
    cal_pot_uterm(dftu, ucell, pv, spin, pot_uterm, sk, nullptr);
}

template <typename T>
void DFTU_LCAO::cal_pot_uterm(Plus_U_Base& dftu,
                              const UnitCell& ucell,
                              const Parallel_Orbitals* pv,
                              const int spin,
                              T* pot_uterm,
                              const T* sk,
                              const ModuleBase::DdlaContext* ddla)
{
    ModuleBase::TITLE("DFTU_LCAO", "cal_pot_uterm");
    ModuleBase::timer::start("DFTU_LCAO", "cal_pot_uterm");
    std::fill(pot_uterm, pot_uterm + pv->para_2d().size(), T(0));
    std::vector<T> onsite(pv->para_2d().size());
    cal_pot_onsite(dftu, ucell, pv, spin, true, onsite.data());
    cal_pot_uterm(onsite, sk, pot_uterm, pv->para_2d(), ddla);
    ModuleBase::timer::end("DFTU_LCAO", "cal_pot_uterm");
}

template <typename T>
void DFTU_LCAO::cal_pot_uterm(std::vector<T>& onsite, const T* overlap, T* output,
                              const BlockCyclicDistribution& dist,
                              const ModuleBase::DdlaContext* ddla)
{
#ifdef __MPI
    const int n = dist.nrglb();
    const T half = T(0.5);
    const T one = T(1.0);
    const T zero = T(0.0);
    const ModuleBase::DdlaContext* selected = ModuleBase::ddla_linalg::bound_backend(ddla, dist);
    if (selected != nullptr)
    {
        using ModuleBase::ddla_linalg::ResidentMatrix;
        const ResidentMatrix<T> v(selected, onsite.data(), dist.desc());
        const ResidentMatrix<T> s(selected, overlap, dist.desc());
        ResidentMatrix<T> product(selected, dist.desc());
        ResidentMatrix<T> result(selected, output, dist.desc(), false);
        ModuleBase::ddla_linalg::gemm('N', 'N', n, n, n, half, v, s, zero, product);
        ModuleBase::ddla_linalg::geadd('T', n, n, one, product, zero, result);
        ModuleBase::ddla_linalg::geadd('N', n, n, one, product, one, result);
        result.copy_to_host(output);
    }
    else
    {
        const int first = 1;
        const char no_transpose = 'N';
        ScalapackConnector::gemm(no_transpose, no_transpose, n, n, n, half,
                                 onsite.data(), first, first, dist.desc(),
                                 overlap, first, first, dist.desc(), zero,
                                 output, first, first, dist.desc());
        for (std::size_t i = 0; i < dist.size(); ++i) { onsite[i] = output[i]; }
        ScalapackConnector::tranu(n, n, one, onsite.data(), first, first, dist.desc(),
                                  one, output, first, first, dist.desc());
    }
#else
    (void)onsite;
    (void)overlap;
    (void)output;
    (void)dist;
    (void)ddla;
#endif
}


// Explicit instantiation
template void DFTU_LCAO::cal_pot_uterm<double>(Plus_U_Base& dftu,
                                               const UnitCell& ucell,
                                               const Parallel_Orbitals* pv,
                                               const int spin,
                                               double* pot_uterm,
                                               const double* sk);
template void DFTU_LCAO::cal_pot_uterm<std::complex<double>>(Plus_U_Base& dftu,
                                                             const UnitCell& ucell,
                                                             const Parallel_Orbitals* pv,
                                                             const int spin,
                                                             std::complex<double>* pot_uterm,
                                                             const std::complex<double>* sk);

template void DFTU_LCAO::cal_pot_uterm<double>(Plus_U_Base&, const UnitCell&,
    const Parallel_Orbitals*, const int, double*, const double*, const ModuleBase::DdlaContext*);
template void DFTU_LCAO::cal_pot_uterm<std::complex<double>>(Plus_U_Base&, const UnitCell&,
    const Parallel_Orbitals*, const int, std::complex<double>*, const std::complex<double>*, const ModuleBase::DdlaContext*);
template void DFTU_LCAO::cal_pot_uterm<double>(std::vector<double>&, const double*, double*,
    const BlockCyclicDistribution&, const ModuleBase::DdlaContext*);
template void DFTU_LCAO::cal_pot_uterm<std::complex<double>>(std::vector<std::complex<double>>&,
    const std::complex<double>*, std::complex<double>*, const BlockCyclicDistribution&, const ModuleBase::DdlaContext*);

double DFTU_LCAO::get_onsite_pot(const Plus_U_Base& dftu,
                                 const int T,
                                 const int iat,
                                 const int L,
                                 const int spin,
                                 const int m0,
                                 const int m1,
                                 const bool new_occ_mat)
{
    ModuleBase::TITLE("DFTU_LCAO", "get_onsite_pot");

    double pot_onsite = 0.0;

    switch (dftu.get_form())
    {
    case Plus_U_Base::UForm::lich_fll: // Lichtenstein (rotationally invariant) + FLL DC
        break;

    case Plus_U_Base::UForm::lich_amf: // Lichtenstein (rotationally invariant) + AMF DC
        break;

    case Plus_U_Base::UForm::dud_fll: // Dudarev (simplified) + FLL DC
        if (new_occ_mat)
        {
            if (dftu.use_yukawa())
            {
                if (m0 == m1)
                {
                    pot_onsite = (dftu.yukawa().get_U(T, L) - dftu.yukawa().get_J(T, L))
                                 * (0.5 - dftu.occmat().get(iat, L, spin, m0, m1));
                }
                else
                {
                    pot_onsite = -(dftu.yukawa().get_U(T, L) - dftu.yukawa().get_J(T, L))
                                 * dftu.occmat().get(iat, L, spin, m0, m1);
                }
            }
            else
            {
                if (m0 == m1)
                {
                    pot_onsite = dftu.get_u_current(T)
                                 * (0.5 - dftu.occmat().get(iat, L, spin, m0, m1));
                }
                else
                {
                    pot_onsite = -dftu.get_u_current(T)
                                 * dftu.occmat().get(iat, L, spin, m0, m1);
                }
            }
        }
        else
        {
            if (dftu.use_yukawa())
            {
                if (m0 == m1)
                {
                    pot_onsite = (dftu.yukawa().get_U(T, L) - dftu.yukawa().get_J(T, L))
                                 * (0.5 - dftu.occmat().get_save(iat, L, spin, m0, m1));
                }
                else
                {
                    pot_onsite = -(dftu.yukawa().get_U(T, L) - dftu.yukawa().get_J(T, L))
                                 * dftu.occmat().get_save(iat, L, spin, m0, m1);
                }
            }
            else
            {
                if (m0 == m1)
                {
                    pot_onsite = dftu.get_u_current(T)
                                 * (0.5 - dftu.occmat().get_save(iat, L, spin, m0, m1));
                }
                else
                {
                    pot_onsite = -dftu.get_u_current(T)
                                 * dftu.occmat().get_save(iat, L, spin, m0, m1);
                }
            }
        }

        break;
    }

    return pot_onsite;
}
