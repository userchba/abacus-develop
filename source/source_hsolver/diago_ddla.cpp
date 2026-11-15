#include "diago_ddla.h"

#ifdef __DDLA
#include "source_base/timer.h"
#include "source_base/tool_quit.h"
#include "source_base/tool_title.h"
#include "source_lcao/module_rt/ddla_context.h"
#include "source_lcao/module_rt/td_ddla_device.h"

#include <ddla/ddla.h>

#include <algorithm>
#include <complex>
#include <exception>
#include <string>
#include <vector>

namespace hsolver
{

template <typename T>
void DiagoDdla<T>::diag(ModuleBase::MatrixBlock<T>& h_mat,
                        ModuleBase::MatrixBlock<T>& s_mat,
                        psi::Psi<T>& psi,
                        Real* eigenvalue_in)
{
    ModuleBase::TITLE("DiagoDdla", "diag");
    ModuleBase::timer::start("DiagoDdla", "diag");

    const ddla::DdlaHandle_t& handle = this->ddla.handle();
    const int* desc = h_mat.desc;
    // psi holds the leading nbands columns of the layout of H.
    int desc_psi[9];
    std::copy(desc, desc + 9, desc_psi);
    desc_psi[3] = this->nbands;

    const char jobz = 'V';
    const char uplo = 'U';
    std::vector<Real> eigen(this->nlocal, 0.0);
    int info = 0;
    try
    {
        // phegvd destroys A and overwrites B with its Cholesky factor, so it works
        // on copies of H and S in the handle's memory (device memory for a GPU
        // handle). The eigenvectors overwrite the copy of H; their leading nbands
        // columns are copied to psi.
        ddla::vector<T> h_work = module_rt::ddla_device::to_device(handle, h_mat.p, desc);
        ddla::vector<T> s_work = module_rt::ddla_device::to_device(handle, s_mat.p, desc);
        ddla::phegvd(handle,
                     jobz,
                     uplo,
                     this->nlocal,
                     this->nbands,
                     h_work.data(),
                     desc,
                     s_work.data(),
                     desc,
                     eigen.data(),
                     h_work.data(),
                     desc,
                     info);
        if (info == 0)
        {
            module_rt::ddla_device::to_host(handle, psi.get_pointer(), h_work, desc_psi);
        }
    }
    catch (const std::exception& e)
    {
        const std::string message = std::string("LibDDLA phegvd failed: ") + e.what();
        ModuleBase::WARNING_QUIT("DiagoDdla::diag", message);
    }
    if (info != 0)
    {
        std::string message = "LibDDLA phegvd returned info = " + std::to_string(info);
        if (info > this->nlocal)
        {
            message += ", the overlap matrix is not positive definite";
        }
        ModuleBase::WARNING_QUIT("DiagoDdla::diag", message);
    }
    std::copy(eigen.begin(), eigen.begin() + this->nbands, eigenvalue_in);

    ModuleBase::timer::end("DiagoDdla", "diag");
}

template class DiagoDdla<double>;
template class DiagoDdla<std::complex<double>>;

} // namespace hsolver

#endif // __DDLA
