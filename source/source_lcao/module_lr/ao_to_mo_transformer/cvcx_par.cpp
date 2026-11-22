#ifdef __MPI
#include "cvcx.h"
#include "source_base/module_external/ddla_linalg.h"
#include "source_base/tool_title.h"
#include "source_lcao/module_lr/utils/lr_util.h"
#include "source_lcao/module_lr/utils/lr_util_print.h"
namespace LR
{
    template <>
    void CVCX_occ_pblas(
        const ModuleBase::DdlaContext* ddla,
        const std::vector<container::Tensor>& V_istate,
        const BlockCyclicDistribution& pmat,
        const psi::Psi<double, base_device::DEVICE_CPU>& c,
        const BlockCyclicDistribution& pc,
        const double* const X_istate,
        const BlockCyclicDistribution& px,
        const int& naos,
        const int& nocc,
        const int& nvirt,
        double* const AX_istate,
        const bool add_on,
        const double factor)
    {
        ModuleBase::TITLE("hamilt_lrtd", "CVCX_occ_pblas");
        assert(pmat.comm() == pc.comm());
        assert(pmat.comm() == px.comm());
        assert(pmat.ictxt() == pc.ictxt());
        assert(pmat.ictxt() == px.ictxt());

        // Empty local blocks still participate in every collective PBLAS call.
        double empty_block(0);
        const double* const X_istate_data = X_istate == nullptr ? &empty_block : X_istate;
        double* const AX_istate_data = AX_istate == nullptr ? &empty_block : AX_istate;

        const int nks = c.get_nk();
        assert(V_istate.size() == nks);

        BlockCyclicDistribution pcv;
        LR_Util::setup_2d_division(pcv, pmat.nb(), nocc, naos, pmat.ictxt());
        BlockCyclicDistribution pcx;
        LR_Util::setup_2d_division(pcx, pmat.nb(), naos, nvirt, pmat.ictxt());
        for (int isk = 0;isk < nks;++isk)
        {
            c.fix_k(isk);
            const int start = isk * px.size();

            const int i1 = 1;
            const int ivirt = nocc + 1;
            const char trans = 'T';
            const char notrans = 'N';  //c is col major
            const double one = 1.0;
            const double zero = 0.0;

            // c^TV[nocc*naos]
            using ModuleBase::ddla_linalg::ResidentMatrix;
            const ResidentMatrix<double> coefficients(ddla, c.get_pointer(), pc.desc());
            const ResidentMatrix<double> potential(ddla, V_istate[isk].data<double>(), pmat.desc());
            const double* const x_input = X_istate_data + start;
            const ResidentMatrix<double> x(ddla, x_input, px.desc());
            ResidentMatrix<double> cv(ddla, pcv.desc());
            ResidentMatrix<double> cx(ddla, pcx.desc());
            double* const output = AX_istate_data + start;
            ResidentMatrix<double> ax(ddla, output, px.desc(), add_on);
            ModuleBase::ddla_linalg::gemm(
                trans,
                notrans,
                nocc,
                naos,
                naos,
                one,
                coefficients,
                i1,
                i1,
                potential,
                i1,
                i1,
                zero,
                cv,
                i1,
                i1);

            // cX^T[naos*nvirt]
            ModuleBase::ddla_linalg::gemm(
                notrans,
                trans,
                naos,
                nvirt,
                nocc,
                one,
                coefficients,
                i1,
                i1,
                x,
                i1,
                i1,
                zero,
                cx,
                i1,
                i1);

            //AX_istate=[cX^T]^T[c^TV]^T (nvirt major)
            const auto accumulation_beta_10 = add_on ? one : zero;
            ModuleBase::ddla_linalg::gemm(
                trans,
                trans,
                nvirt,
                nocc,
                naos,
                factor,
                cx,
                i1,
                i1,
                cv,
                i1,
                i1,
                accumulation_beta_10,
                ax,
                i1,
                i1);
            ax.copy_to_host(output);
        }
    }

    template <>
    void CVCX_occ_pblas(
        const ModuleBase::DdlaContext* ddla,
        const std::vector<container::Tensor>& V_istate,
        const BlockCyclicDistribution& pmat,
        const psi::Psi<std::complex<double>, base_device::DEVICE_CPU>& c,
        const BlockCyclicDistribution& pc,
        const std::complex<double>* const X_istate,
        const BlockCyclicDistribution& px,
        const int& naos,
        const int& nocc,
        const int& nvirt,
        std::complex<double>* const AX_istate,
        const bool add_on,
        const std::complex<double> factor)
    {
        ModuleBase::TITLE("hamilt_lrtd", "CVCX_occ_pblas");
        assert(pmat.comm() == pc.comm());
        assert(pmat.comm() == px.comm());
        assert(pmat.ictxt() == pc.ictxt());
        assert(pmat.ictxt() == px.ictxt());

        // Empty local blocks still participate in every collective PBLAS call.
        std::complex<double> empty_block(0);
        const std::complex<double>* const X_istate_data = X_istate == nullptr ? &empty_block : X_istate;
        std::complex<double>* const AX_istate_data = AX_istate == nullptr ? &empty_block : AX_istate;

        int nks = c.get_nk();
        assert(V_istate.size() == nks);

        BlockCyclicDistribution pcv;
        LR_Util::setup_2d_division(pcv, pmat.nb(), nocc, naos, pmat.ictxt());
        BlockCyclicDistribution pcx;
        LR_Util::setup_2d_division(pcx, pmat.nb(), naos, nvirt, pmat.ictxt());
        for (int isk = 0;isk < nks;++isk)
        {
            c.fix_k(isk);
            const int start = isk * px.size();

            const int i1 = 1;
            const int ivirt = nocc + 1;
            const char trans = 'T';
            const char dagger = 'C';
            const char notrans = 'N';  //c is col major
            const std::complex<double> one(1.0, 0.0);
            const std::complex<double> zero(0.0, 0.0);

            // c^TV[nocc*naos]
            using ModuleBase::ddla_linalg::ResidentMatrix;
            const ResidentMatrix<std::complex<double>> coefficients(ddla, c.get_pointer(), pc.desc());
            const ResidentMatrix<std::complex<double>> potential(ddla, V_istate[isk].data<std::complex<double>>(), pmat.desc());
            const std::complex<double>* const x_input = X_istate_data + start;
            const ResidentMatrix<std::complex<double>> x(ddla, x_input, px.desc());
            ResidentMatrix<std::complex<double>> cv(ddla, pcv.desc());
            ResidentMatrix<std::complex<double>> cx(ddla, pcx.desc());
            std::complex<double>* const output = AX_istate_data + start;
            ResidentMatrix<std::complex<double>> ax(ddla, output, px.desc(), add_on);
            ModuleBase::ddla_linalg::gemm(
                dagger,
                notrans,
                nocc,
                naos,
                naos,
                one,
                coefficients,
                i1,
                i1,
                potential,
                i1,
                i1,
                zero,
                cv,
                i1,
                i1);

            // cX^T[naos*nvirt]
            ModuleBase::ddla_linalg::gemm(
                notrans,
                dagger,
                naos,
                nvirt,
                nocc,
                one,
                coefficients,
                i1,
                i1,
                x,
                i1,
                i1,
                zero,
                cx,
                i1,
                i1);

            //AX_istate=[cX^T]^T[c^TV]^T (nvirt major)
            const auto accumulation_beta_7 = add_on ? one : zero;
            ModuleBase::ddla_linalg::gemm(
                trans,
                trans,
                nvirt,
                nocc,
                naos,
                factor,
                cx,
                i1,
                i1,
                cv,
                i1,
                i1,
                accumulation_beta_7,
                ax,
                i1,
                i1);
            ax.copy_to_host(output);
        }
    }

    template <>
    void CVCX_virt_pblas(
        const ModuleBase::DdlaContext* ddla,
        const std::vector<container::Tensor>& V_istate,
        const BlockCyclicDistribution& pmat,
        const psi::Psi<double, base_device::DEVICE_CPU>& c,
        const BlockCyclicDistribution& pc,
        const double* const X_istate,
        const BlockCyclicDistribution& px,
        const int& naos,
        const int& nocc,
        const int& nvirt,
        double* const AX_istate,
        const bool add_on,
        const double factor)
    {
        ModuleBase::TITLE("hamilt_lrtd", "CVCX_virt_pblas");
        assert(pmat.comm() == pc.comm());
        assert(pmat.comm() == px.comm());
        assert(pmat.ictxt() == pc.ictxt());
        assert(pmat.ictxt() == px.ictxt());

        // Empty local blocks still participate in every collective PBLAS call.
        double empty_block(0);
        const double* const X_istate_data = X_istate == nullptr ? &empty_block : X_istate;
        double* const AX_istate_data = AX_istate == nullptr ? &empty_block : AX_istate;

        const int nks = c.get_nk();
        assert(V_istate.size() == nks);

        BlockCyclicDistribution pcv;
        LR_Util::setup_2d_division(pcv, pmat.nb(), naos, nvirt, pmat.ictxt());
        BlockCyclicDistribution pcx;
        LR_Util::setup_2d_division(pcx, pmat.nb(), nocc, naos, pmat.ictxt());
        for (int isk = 0;isk < nks;++isk)
        {
            c.fix_k(isk);
            const int start = isk * px.size();

            const int i1 = 1;
            const int ivirt = nocc + 1;
            const char trans = 'T';
            const char dagger = 'C';
            const char notrans = 'N';  //c is col major
            const double one = 1.0;
            const double zero = 0.0;

            // VC[naos*nvirt]
            using ModuleBase::ddla_linalg::ResidentMatrix;
            const ResidentMatrix<double> coefficients(ddla, c.get_pointer(), pc.desc());
            const ResidentMatrix<double> potential(ddla, V_istate[isk].data<double>(), pmat.desc());
            const double* const x_input = X_istate_data + start;
            const ResidentMatrix<double> x(ddla, x_input, px.desc());
            ResidentMatrix<double> cv(ddla, pcv.desc());
            ResidentMatrix<double> cx(ddla, pcx.desc());
            double* const output = AX_istate_data + start;
            ResidentMatrix<double> ax(ddla, output, px.desc(), add_on);
            ModuleBase::ddla_linalg::gemm(
                notrans,
                notrans,
                naos,
                nvirt,
                naos,
                one,
                potential,
                i1,
                i1,
                coefficients,
                i1,
                ivirt,
                zero,
                cv,
                i1,
                i1);

            // X^TC^T[nocc*naos]
            ModuleBase::ddla_linalg::gemm(
                dagger,
                dagger,
                nocc,
                naos,
                nvirt,
                one,
                x,
                i1,
                i1,
                coefficients,
                i1,
                ivirt,
                zero,
                cx,
                i1,
                i1);

            //AX_istate=[VC]^T[X^TC^T]^T (nvirt major)
            const auto accumulation_beta_4 = add_on ? one : zero;
            ModuleBase::ddla_linalg::gemm(
                trans,
                trans,
                nvirt,
                nocc,
                naos,
                factor,
                cv,
                i1,
                i1,
                cx,
                i1,
                i1,
                accumulation_beta_4,
                ax,
                i1,
                i1);
            ax.copy_to_host(output);
        }
    }

    template <>
    void CVCX_virt_pblas(
        const ModuleBase::DdlaContext* ddla,
        const std::vector<container::Tensor>& V_istate,
        const BlockCyclicDistribution& pmat,
        const psi::Psi<std::complex<double>, base_device::DEVICE_CPU>& c,
        const BlockCyclicDistribution& pc,
        const std::complex<double>* const X_istate,
        const BlockCyclicDistribution& px,
        const int& naos,
        const int& nocc,
        const int& nvirt,
        std::complex<double>* const AX_istate,
        const bool add_on,
        const std::complex<double> factor)
    {
        ModuleBase::TITLE("hamilt_lrtd", "CVCX_virt_pblas");
        assert(pmat.comm() == pc.comm());
        assert(pmat.comm() == px.comm());
        assert(pmat.ictxt() == pc.ictxt());
        assert(pmat.ictxt() == px.ictxt());

        // Empty local blocks still participate in every collective PBLAS call.
        std::complex<double> empty_block(0);
        const std::complex<double>* const X_istate_data = X_istate == nullptr ? &empty_block : X_istate;
        std::complex<double>* const AX_istate_data = AX_istate == nullptr ? &empty_block : AX_istate;

        const int nks = c.get_nk();
        assert(V_istate.size() == nks);

        BlockCyclicDistribution pcv;
        LR_Util::setup_2d_division(pcv, pmat.nb(), naos, nvirt, pmat.ictxt());
        BlockCyclicDistribution pcx;
        LR_Util::setup_2d_division(pcx, pmat.nb(), nocc, naos, pmat.ictxt());
        for (int isk = 0;isk < nks;++isk)
        {
            c.fix_k(isk);
            const int start = isk * px.size();

            const int i1 = 1;
            const int ivirt = nocc + 1;
            const char trans = 'T';
            const char dagger = 'C';
            const char notrans = 'N';  //c is col major
            const std::complex<double> one(1.0, 0.0);
            const std::complex<double> zero(0.0, 0.0);

            // VC[naos*nvirt]
            using ModuleBase::ddla_linalg::ResidentMatrix;
            const ResidentMatrix<std::complex<double>> coefficients(ddla, c.get_pointer(), pc.desc());
            const ResidentMatrix<std::complex<double>> potential(ddla, V_istate[isk].data<std::complex<double>>(), pmat.desc());
            const std::complex<double>* const x_input = X_istate_data + start;
            const ResidentMatrix<std::complex<double>> x(ddla, x_input, px.desc());
            ResidentMatrix<std::complex<double>> cv(ddla, pcv.desc());
            ResidentMatrix<std::complex<double>> cx(ddla, pcx.desc());
            std::complex<double>* const output = AX_istate_data + start;
            ResidentMatrix<std::complex<double>> ax(ddla, output, px.desc(), add_on);
            ModuleBase::ddla_linalg::gemm(
                notrans,
                notrans,
                naos,
                nvirt,
                naos,
                one,
                potential,
                i1,
                i1,
                coefficients,
                i1,
                ivirt,
                zero,
                cv,
                i1,
                i1);

            // X^TC^T[nocc*naos]
            ModuleBase::ddla_linalg::gemm(
                dagger,
                dagger,
                nocc,
                naos,
                nvirt,
                one,
                x,
                i1,
                i1,
                coefficients,
                i1,
                ivirt,
                zero,
                cx,
                i1,
                i1);

            //AX_istate=[VC]^T[X^TC^T]^T (nvirt major)
            const auto accumulation_beta_1 = add_on ? one : zero;
            ModuleBase::ddla_linalg::gemm(
                trans,
                trans,
                nvirt,
                nocc,
                naos,
                factor,
                cv,
                i1,
                i1,
                cx,
                i1,
                i1,
                accumulation_beta_1,
                ax,
                i1,
                i1);
            ax.copy_to_host(output);
        }
    }
}
#endif