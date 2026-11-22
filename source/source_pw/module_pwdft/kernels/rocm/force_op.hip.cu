#include "source_pw/module_pwdft/kernels/force_op.h"

#include <complex>

#include <thrust/complex.h>
#include <hip/hip_runtime.h>
#include <base/macros/macros.h>

#define THREADS_PER_BLOCK 256

namespace hamilt {

template <typename FPTYPE>
__global__ void cal_vkb1_nl(
        const int npwx,
        const int vkb_nc,
        const int nbasis,
        const int ipol,
        const thrust::complex<FPTYPE> NEG_IMAG_UNIT,
        const thrust::complex<FPTYPE> *vkb,
        const FPTYPE *gcar,
        thrust::complex<FPTYPE> *vkb1)
{
    thrust::complex<FPTYPE> *pvkb1 = vkb1 + blockIdx.x * npwx;
    const thrust::complex<FPTYPE> *pvkb = vkb + blockIdx.x * vkb_nc;
    for (int ig = threadIdx.x; ig < nbasis; ig += blockDim.x) {
        pvkb1[ig] = pvkb[ig] * NEG_IMAG_UNIT * gcar[ig * 3 + ipol];
    }
}

template <typename FPTYPE>
__global__ void cal_force_nl(
        const bool nondiagonal,
        const int ntype,
        const int spin,
        const int deeq_2,
        const int deeq_3,
        const int deeq_4,
        const int forcenl_nc,
        const int nbands,
        const int nkb,
        const int *atom_nh,
        const int *atom_na,
        const FPTYPE tpiba,
        const FPTYPE *d_wg,
        const bool occ,
        const FPTYPE* d_ekb,
        const FPTYPE* qq_nt,
        const FPTYPE *deeq,
        const thrust::complex<FPTYPE> *becp,
        const thrust::complex<FPTYPE> *dbecp,
        FPTYPE *force)
{
    const int ib = blockIdx.x / ntype;
    const int it = blockIdx.x % ntype;

    int iat = 0;
    int sum = 0;
    for (int ii = 0; ii < it; ii++) {
        iat += atom_na[ii];
        sum += atom_na[ii] * atom_nh[ii];
    }

    int nproj = atom_nh[it];
    FPTYPE fac;
    if(occ)
    {
        fac = d_wg[ib] * 2.0 * tpiba;
    }
    else
    {
        fac = d_wg[0] * 2.0 * tpiba;
    }
    FPTYPE ekb_now = 0.0;
    if (d_ekb != nullptr)
    {
        ekb_now = d_ekb[ib];
    }
    for (int ia = 0; ia < atom_na[it]; ia++) {
        for (int ip = threadIdx.x; ip < nproj; ip += blockDim.x) {
            FPTYPE ps_qq = 0;
            if(ekb_now != 0)
            {
                ps_qq = - ekb_now * qq_nt[it * deeq_3 * deeq_4 + ip * deeq_4 + ip];
            }
            FPTYPE ps = deeq[((spin * deeq_2 + iat) * deeq_3 + ip) * deeq_4 + ip] + ps_qq;
            const int inkb = sum + ip;
            //out<<"\n ps = "<<ps;

            for (int ipol = 0; ipol < 3; ipol++) {
                const FPTYPE dbb = (conj(dbecp[ipol * nbands * nkb + ib * nkb + inkb]) *
                                    becp[ib * nkb + inkb]).real();
                // force[iat * forcenl_nc + ipol] -= ps * fac * dbb;
                atomicAdd(force + iat * forcenl_nc + ipol, -ps * fac * dbb);
                //cf[iat*3+ipol] += ps * fac * dbb;
            }

            if (nondiagonal) {
                //for (int ip2=0; ip2<nproj; ip2++)
                for (int ip2 = 0; ip2 < nproj; ip2++) {
                    if (ip != ip2) {
                        const int jnkb = sum + ip2;
                        FPTYPE ps_qq = 0;
                        if(ekb_now != 0)
                        {
                            ps_qq = - ekb_now * qq_nt[it * deeq_3 * deeq_4 + ip * deeq_4 + ip2];
                        }
                        ps = deeq[((spin * deeq_2 + iat) * deeq_3 + ip) * deeq_4 + ip2] + ps_qq;
                        for (int ipol = 0; ipol < 3; ipol++) {
                            const FPTYPE dbb = (conj(dbecp[ipol * nbands * nkb + ib * nkb + inkb]) *
                                                becp[ib * nkb + jnkb]).real();
                            atomicAdd(force + iat * forcenl_nc + ipol, -ps * fac * dbb);
                        }
                    }
                }
            }
        }
        iat += 1;
        sum += nproj;
    }
}

template <typename FPTYPE>
void cal_vkb1_nl_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                                 const int& nkb,
                                                                 const int& npwx,
                                                                 const int& vkb_nc,
                                                                 const int& nbasis,
                                                                 const int& ipol,
                                                                 const std::complex<FPTYPE>& NEG_IMAG_UNIT,
                                                                 const std::complex<FPTYPE>* vkb,
                                                                 const FPTYPE* gcar,
                                                                 std::complex<FPTYPE>* vkb1)
{
    hipLaunchKernelGGL(HIP_KERNEL_NAME(cal_vkb1_nl<FPTYPE>), dim3(nkb), dim3(THREADS_PER_BLOCK), 0, 0,
            npwx,
            vkb_nc,
            nbasis,
            ipol,
            static_cast<const thrust::complex<FPTYPE>>(NEG_IMAG_UNIT), // array of data
            reinterpret_cast<const thrust::complex<FPTYPE>*>(vkb),
            gcar,// array of data
            reinterpret_cast<thrust::complex<FPTYPE>*>(vkb1)); // array of data

    hipCheckOnDebug();
}

template <typename FPTYPE>
void cal_force_nl_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                                  const bool& nondiagonal,
                                                                  const int& nbands_occ,
                                                                  const int& ntype,
                                                                  const int& spin,
                                                                  const int& deeq_2,
                                                                  const int& deeq_3,
                                                                  const int& deeq_4,
                                                                  const int& forcenl_nc,
                                                                  const int& nbands,
                                                                  const int& nkb,
                                                                  const int* atom_nh,
                                                                  const int* atom_na,
                                                                  const FPTYPE& tpiba,
                                                                  const FPTYPE* d_wg,
                                                                  const bool& occ,
                                                                  const FPTYPE* d_ekb,
                                                                  const FPTYPE* qq_nt,
                                                                  const FPTYPE* deeq,
                                                                  const std::complex<FPTYPE>* becp,
                                                                  const std::complex<FPTYPE>* dbecp,
                                                                  FPTYPE* force)
{
    hipLaunchKernelGGL(HIP_KERNEL_NAME(cal_force_nl<FPTYPE>), dim3(nbands_occ * ntype), dim3(THREADS_PER_BLOCK), 0, 0,
            nondiagonal,
            ntype, spin,
            deeq_2, deeq_3, deeq_4,
            forcenl_nc, nbands, nkb,
            atom_nh, atom_na,
            tpiba,
            d_wg, occ, d_ekb, qq_nt, deeq,
            reinterpret_cast<const thrust::complex<FPTYPE>*>(becp),
            reinterpret_cast<const thrust::complex<FPTYPE>*>(dbecp),
            force);// array of data

    hipCheckOnDebug();
}

template <typename FPTYPE>
__global__ void cal_force_nl(
        const int ntype,
        const int deeq_2,
        const int deeq_3,
        const int deeq_4,
        const int forcenl_nc,
        const int nbands,
        const int nkb,
        const int *atom_nh,
        const int *atom_na,
        const FPTYPE tpiba,
        const FPTYPE *d_wg,
        const bool occ,
        const FPTYPE* d_ekb,
        const FPTYPE* qq_nt,
        const thrust::complex<FPTYPE> *deeq_nc,
        const thrust::complex<FPTYPE> *becp,
        const thrust::complex<FPTYPE> *dbecp,
        FPTYPE *force)
{
    const int ib = blockIdx.x / ntype;
    const int ib2 = ib * 2;
    const int it = blockIdx.x % ntype;

    int iat = 0;
    int sum = 0;
    for (int ii = 0; ii < it; ii++) {
        iat += atom_na[ii];
        sum += atom_na[ii] * atom_nh[ii];
    }

    int nproj = atom_nh[it];
    FPTYPE fac;
    if(occ)
    {
        fac = d_wg[ib] * 2.0 * tpiba;
    }
    else
    {
        fac = d_wg[0] * 2.0 * tpiba;
    }
    FPTYPE ekb_now = 0.0;
    if (d_ekb != nullptr)
    {
        ekb_now = d_ekb[ib];
    }
    for (int ia = 0; ia < atom_na[it]; ia++) {
        for (int ip = threadIdx.x; ip < nproj; ip += blockDim.x) {
            const int inkb = sum + ip;
            for (int ip2 = 0; ip2 < nproj; ip2++) 
            {
                // Effective values of the D-eS coefficients
                thrust::complex<FPTYPE> ps_qq = 0;
                if (ekb_now)
                {
                    ps_qq = thrust::complex<FPTYPE>(-ekb_now * qq_nt[it * deeq_3 * deeq_4 + ip * deeq_4 + ip2], 0.0);
                }
                const int jnkb = sum + ip2;
                const thrust::complex<FPTYPE> ps0 = deeq_nc[((0 * deeq_2 + iat) * deeq_3 + ip) * deeq_4 + ip2] + ps_qq;
                const thrust::complex<FPTYPE> ps1 = deeq_nc[((1 * deeq_2 + iat) * deeq_3 + ip) * deeq_4 + ip2];
                const thrust::complex<FPTYPE> ps2 = deeq_nc[((2 * deeq_2 + iat) * deeq_3 + ip) * deeq_4 + ip2];
                const thrust::complex<FPTYPE> ps3 = deeq_nc[((3 * deeq_2 + iat) * deeq_3 + ip) * deeq_4 + ip2] + ps_qq;

                for (int ipol = 0; ipol < 3; ipol++) {
                    const int index0 = ipol * nbands * 2 * nkb + ib2 * nkb + inkb;
                    const int index1 = ib2 * nkb + jnkb;
                    const thrust::complex<FPTYPE> dbb0 = conj(dbecp[index0]) * becp[index1];
                    const thrust::complex<FPTYPE> dbb1 = conj(dbecp[index0]) * becp[index1 + nkb];
                    const thrust::complex<FPTYPE> dbb2 = conj(dbecp[index0 + nkb]) * becp[index1];
                    const thrust::complex<FPTYPE> dbb3 = conj(dbecp[index0 + nkb]) * becp[index1 + nkb];
                    const FPTYPE tmp = - fac * (ps0 * dbb0 + ps1 * dbb1 + ps2 * dbb2 + ps3 * dbb3).real();
                    atomicAdd(force + iat * forcenl_nc + ipol, tmp);
                }
            }
        }
        iat += 1;
        sum += nproj;
    }
}

// interface for nspin=4 only
template <typename FPTYPE>
void cal_force_nl_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                    const int& nbands_occ,
                    const int& ntype,
                    const int& deeq_2,
                    const int& deeq_3,
                    const int& deeq_4,
                    const int& forcenl_nc,
                    const int& nbands,
                    const int& nkb,
                    const int* atom_nh,
                    const int* atom_na,
                    const FPTYPE& tpiba,
                    const FPTYPE* d_wg,
                    const bool& occ,
                    const FPTYPE* d_ekb,
                    const FPTYPE* qq_nt,
                    const std::complex<FPTYPE>* deeq_nc,
                    const std::complex<FPTYPE>* becp,
                    const std::complex<FPTYPE>* dbecp,
                    FPTYPE* force)
{
    hipLaunchKernelGGL(HIP_KERNEL_NAME(cal_force_nl<FPTYPE>), dim3(nbands_occ * ntype), dim3(THREADS_PER_BLOCK), 0, 0,
            ntype,
            deeq_2, deeq_3, deeq_4,
            forcenl_nc, nbands, nkb,
            atom_nh, atom_na,
            tpiba,
            d_wg, occ, d_ekb, qq_nt, 
            reinterpret_cast<const thrust::complex<FPTYPE>*>(deeq_nc),
            reinterpret_cast<const thrust::complex<FPTYPE>*>(becp),
            reinterpret_cast<const thrust::complex<FPTYPE>*>(dbecp),
            force);// array of data

    hipCheckOnDebug();
}

template <typename FPTYPE, int npol>
__global__ void cal_force_onsite(int wg_nc,
                                  int ntype,
                                  int forcenl_nc,
                                  int nbands,
                                  int ik,
                                  int nkb,
                                  const int* atom_nh,
                                  const int* atom_na,
                                  FPTYPE tpiba,
                                  const FPTYPE* d_wg,
                                  const thrust::complex<FPTYPE>* pot_onsite,
                                  const int* l_channel,
                                  const thrust::complex<FPTYPE>* becp,
                                  const thrust::complex<FPTYPE>* dbecp,
                                  FPTYPE* force)
{
    const int ib = blockIdx.x / ntype;
    const int it = blockIdx.x % ntype;
    if (l_channel[it] == -1)
        return;
    const int orbital_l = l_channel[it];
    const int ip_begin = orbital_l * orbital_l;
    const int tlp1 = 2 * orbital_l + 1;
    const int tlp1_2 = tlp1 * tlp1;

    int iat = 0;
    int sum = 0;
    for (int ii = 0; ii < it; ii++)
    {
        iat += atom_na[ii];
        sum += atom_na[ii] * atom_nh[ii];
        pot_onsite += npol * npol * tlp1_2 * atom_na[ii];
    }

    const int ib2 = ib * npol;
    const FPTYPE fac = d_wg[ik * wg_nc + ib] * 2.0 * tpiba;
    const int nprojs = atom_nh[it];
    for (int ia = 0; ia < atom_na[it]; ia++)
    {
        for (int mm = threadIdx.x; mm < tlp1_2; mm += blockDim.x)
        {
            const int m1 = mm / tlp1;
            const int m2 = mm % tlp1;
            const int ip1 = ip_begin + m1;
            const int ip2 = ip_begin + m2;
            const int inkb1 = sum + ip1 + ib2 * nkb;
            const int inkb2 = sum + ip2 + ib2 * nkb;
            for (int ipol = 0; ipol < 3; ipol++)
            {
                const int inkb0 = ipol * nbands * npol * nkb + inkb1;
                FPTYPE tmp = 0;
                if (npol == 2)
                {
                    thrust::complex<FPTYPE> ps[4] = {pot_onsite[mm], pot_onsite[mm + tlp1_2], pot_onsite[mm + 2 * tlp1_2], pot_onsite[mm + 3 * tlp1_2]};
                    const thrust::complex<FPTYPE> dbb0 = conj(dbecp[inkb0]) * becp[inkb2];
                    const thrust::complex<FPTYPE> dbb1 = conj(dbecp[inkb0]) * becp[inkb2 + nkb];
                    const thrust::complex<FPTYPE> dbb2 = conj(dbecp[inkb0 + nkb]) * becp[inkb2];
                    const thrust::complex<FPTYPE> dbb3 = conj(dbecp[inkb0 + nkb]) * becp[inkb2 + nkb];
                    tmp = -fac * (ps[0] * dbb0 + ps[1] * dbb1 + ps[2] * dbb2 + ps[3] * dbb3).real();
                }
                else
                {
                    tmp = -fac * (pot_onsite[mm] * conj(dbecp[inkb0]) * becp[inkb2]).real();
                }
                atomicAdd(force + iat * forcenl_nc + ipol, tmp);
            }
        }
        ++iat;
        sum += nprojs;
        pot_onsite += npol * npol * tlp1_2;
    }
}

template <typename FPTYPE, int npol>
__global__ void cal_force_onsite(int wg_nc,
                                 int ntype,
                                 int forcenl_nc,
                                 int nbands,
                                 int ik,
                                 int nkb,
                                 int spin_sign,
                                 const int* atom_nh,
                                 const int* atom_na,
                                 FPTYPE tpiba,
                                 const FPTYPE* d_wg,
                                 const FPTYPE* lambda,
                                 const thrust::complex<FPTYPE>* becp,
                                 const thrust::complex<FPTYPE>* dbecp,
                                 FPTYPE* force)
{
    const int ib = blockIdx.x / ntype;
    const int it = blockIdx.x % ntype;

    int iat = 0;
    int sum = 0;
    for (int ii = 0; ii < it; ii++)
    {
        iat += atom_na[ii];
        sum += atom_na[ii] * atom_nh[ii];
    }

    const int ib2 = ib * npol;
    const FPTYPE fac = d_wg[ik * wg_nc + ib] * 2.0 * tpiba;
    const int nprojs = atom_nh[it];
    for (int ia = 0; ia < atom_na[it]; ia++)
    {
        if (npol == 2)
        {
            const thrust::complex<FPTYPE> coefficients0(lambda[iat * 3 + 2], 0.0);
            // dbb1 = conj(d becp_up) * becp_down requires H_up,down = lambda_x - i lambda_y.
            const thrust::complex<FPTYPE> coefficients1(lambda[iat * 3], -lambda[iat * 3 + 1]);
            const thrust::complex<FPTYPE> coefficients2(lambda[iat * 3], lambda[iat * 3 + 1]);
            const thrust::complex<FPTYPE> coefficients3(-1 * lambda[iat * 3 + 2], 0.0);
            for (int ip = threadIdx.x; ip < nprojs; ip += blockDim.x)
            {
                const int inkb = sum + ip + ib2 * nkb;
                for (int ipol = 0; ipol < 3; ipol++)
                {
                    const int inkb0 = ipol * nbands * 2 * nkb + inkb;
                    const thrust::complex<FPTYPE> dbb0 = conj(dbecp[inkb0]) * becp[inkb];
                    const thrust::complex<FPTYPE> dbb1 = conj(dbecp[inkb0]) * becp[inkb + nkb];
                    const thrust::complex<FPTYPE> dbb2 = conj(dbecp[inkb0 + nkb]) * becp[inkb];
                    const thrust::complex<FPTYPE> dbb3 = conj(dbecp[inkb0 + nkb]) * becp[inkb + nkb];
                    const FPTYPE tmp
                        = -fac
                          * (coefficients0 * dbb0 + coefficients1 * dbb1 + coefficients2 * dbb2 + coefficients3 * dbb3)
                                .real();
                    atomicAdd(force + iat * forcenl_nc + ipol, tmp);
                }
            }
        }
        else
        {
            const FPTYPE lambda_z = lambda[iat * 3 + 2];
            for (int ip = threadIdx.x; ip < nprojs; ip += blockDim.x)
            {
                const int inkb = sum + ip + ib2 * nkb;
                for (int ipol = 0; ipol < 3; ipol++)
                {
                    const int inkb0 = ipol * nbands * nkb + inkb;
                    const FPTYPE dbb = (conj(dbecp[inkb0]) * becp[inkb]).real();
                    atomicAdd(force + iat * forcenl_nc + ipol, -fac * lambda_z * spin_sign * dbb);
                }
            }
        }
        ++iat;
        sum += nprojs;
    }
}

// kernel for DFTU force
template <typename FPTYPE>
void cal_force_nl_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                                   const int& nbands_occ,
                                                                   const int& wg_nc,
                                                                   const int& ntype,
                                                                   const int& forcenl_nc,
                                                                   const int& nbands,
                                                                   const int& ik,
                                                                   const int& nkb,
                                                                   const int& npol,
                                                                   const int* atom_nh,
                                                                   const int* atom_na,
                                                                   const FPTYPE& tpiba,
                                                                   const FPTYPE* d_wg,
                                                                   const std::complex<FPTYPE>* pot_onsite,
                                                                   const int* l_channel,
                                                                   const std::complex<FPTYPE>* becp,
                                                                   const std::complex<FPTYPE>* dbecp,
                                                                   FPTYPE* force)
{
    if (npol == 1)
    {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(cal_force_onsite<FPTYPE, 1>),
                           dim3(nbands_occ * ntype),
                           dim3(THREADS_PER_BLOCK),
                           0,
                           0,
                           wg_nc,
                           ntype,
                           forcenl_nc,
                           nbands,
                           ik,
                           nkb,
                           atom_nh,
                           atom_na,
                           tpiba,
                           d_wg,
                           reinterpret_cast<const thrust::complex<FPTYPE>*>(pot_onsite),
                           l_channel,
                           reinterpret_cast<const thrust::complex<FPTYPE>*>(becp),
                           reinterpret_cast<const thrust::complex<FPTYPE>*>(dbecp),
                           force);
    }
    else
    {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(cal_force_onsite<FPTYPE, 2>),
                           dim3(nbands_occ * ntype),
                           dim3(THREADS_PER_BLOCK),
                           0,
                           0,
                           wg_nc,
                           ntype,
                           forcenl_nc,
                           nbands,
                           ik,
                           nkb,
                           atom_nh,
                           atom_na,
                           tpiba,
                           d_wg,
                           reinterpret_cast<const thrust::complex<FPTYPE>*>(pot_onsite),
                           l_channel,
                           reinterpret_cast<const thrust::complex<FPTYPE>*>(becp),
                           reinterpret_cast<const thrust::complex<FPTYPE>*>(dbecp),
                           force);
    }

    hipCheckOnDebug();
}
// kernel for DeltaSpin force
template <typename FPTYPE>
void cal_force_nl_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                                   const int& nbands_occ,
                                                                   const int& wg_nc,
                                                                   const int& ntype,
                                                                   const int& forcenl_nc,
                                                                   const int& nbands,
                                                                   const int& ik,
                                                                   const int& nkb,
                                                                   const int& npol,
                                                                   const int* atom_nh,
                                                                   const int* atom_na,
                                                                   const FPTYPE& tpiba,
                                                                   const FPTYPE* d_wg,
                                                                   const FPTYPE* lambda,
                                                                   const int* isk,
                                                                   const std::complex<FPTYPE>* becp,
                                                                   const std::complex<FPTYPE>* dbecp,
                                                                   FPTYPE* force)
{
    int spin_sign = 1;
    if (isk != nullptr && npol == 1)
    {
        spin_sign = (isk[ik] == 0) ? 1 : -1;
    }
    if (npol == 1)
    {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(cal_force_onsite<FPTYPE, 1>),
                           dim3(nbands_occ * ntype),
                           dim3(THREADS_PER_BLOCK),
                           0,
                           0,
                           wg_nc,
                           ntype,
                           forcenl_nc,
                           nbands,
                           ik,
                           nkb,
                           spin_sign,
                           atom_nh,
                           atom_na,
                           tpiba,
                           d_wg,
                           lambda,
                           reinterpret_cast<const thrust::complex<FPTYPE>*>(becp),
                           reinterpret_cast<const thrust::complex<FPTYPE>*>(dbecp),
                           force);
    }
    else
    {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(cal_force_onsite<FPTYPE, 2>),
                           dim3(nbands_occ * ntype),
                           dim3(THREADS_PER_BLOCK),
                           0,
                           0,
                           wg_nc,
                           ntype,
                           forcenl_nc,
                           nbands,
                           ik,
                           nkb,
                           spin_sign,
                           atom_nh,
                           atom_na,
                           tpiba,
                           d_wg,
                           lambda,
                           reinterpret_cast<const thrust::complex<FPTYPE>*>(becp),
                           reinterpret_cast<const thrust::complex<FPTYPE>*>(dbecp),
                           force);
    }

    hipCheckOnDebug();
}

template <typename FPTYPE>
__global__ void saveVkbValues_(
    const int *gcar_zero_ptrs, 
    const thrust::complex<FPTYPE> *vkb_ptr, 
    thrust::complex<FPTYPE> *vkb_save_ptr, 
    int nkb,  
    int npw,  
    int ipol,
    int npwx)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x; // global index
    int n_total_gcar_zeros = gcar_zero_ptrs[ipol * npwx];
    const int* gcar_zero_ptr = gcar_zero_ptrs + ipol * npwx + 1; // skip the first element
    int ikb = index / n_total_gcar_zeros;              // index of nkb
    int icount = index % n_total_gcar_zeros;           // index of n_total_gcar_zeros
    
    // check if the index is valid
    if(ikb < nkb )
    {
        int ig = gcar_zero_ptr[icount]; // get ig from gcar_zero_ptrs
        // use the flat index to get the saved position, pay attention to the relationship between ikb and npw,
        vkb_save_ptr[index] = vkb_ptr[ikb * npw + ig];    // save the value
    }
}

template <typename FPTYPE>
void saveVkbValues(
    const int *gcar_zero_ptrs, 
    const std::complex<FPTYPE> *vkb_ptr, 
    std::complex<FPTYPE> *vkb_save_ptr, 
    int nkb, 
    int gcar_zero_count,
    int npw,  
    int ipol,
    int npwx 
    )
{
    hipLaunchKernelGGL(HIP_KERNEL_NAME(saveVkbValues_<FPTYPE>), dim3(nkb*gcar_zero_count), dim3(THREADS_PER_BLOCK), 0, 0,
        gcar_zero_ptrs, 
        reinterpret_cast<const thrust::complex<FPTYPE>*>(vkb_ptr), 
        reinterpret_cast<thrust::complex<FPTYPE>*>(vkb_save_ptr), 
        nkb,  
        npw,  
        ipol,
        npwx);
    hipCheckOnDebug();
}

template <typename FPTYPE>
__global__ void revertVkbValues_(
    const int *gcar_zero_ptrs, 
    thrust::complex<FPTYPE> *vkb_ptr, 
    const thrust::complex<FPTYPE> *vkb_save_ptr, 
    int nkb, 
    int npw,  
    int ipol,
    int npwx,
    const thrust::complex<FPTYPE> coeff)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x; // global index
    int n_total_gcar_zeros = gcar_zero_ptrs[ipol * npwx];
    const int* gcar_zero_ptr = gcar_zero_ptrs + ipol * npwx + 1; // skip the first element
    int ikb = index / n_total_gcar_zeros;              // index of nkb
    int icount = index % n_total_gcar_zeros;           // index of n_total_gcar_zeros
    
    // check if the index is valid
    if(ikb < nkb)
    {
        int ig = gcar_zero_ptr[icount]; // get ig from gcar_zero_ptrs
        // use the flat index to get the saved position, pay attention to the relationship between ikb and npw,
        vkb_ptr[ikb * npw + ig] = vkb_save_ptr[index] * coeff;    // revert the values
    }
}

template <typename FPTYPE>
void revertVkbValues(
    const int *gcar_zero_ptrs, 
    std::complex<FPTYPE> *vkb_ptr, 
    const std::complex<FPTYPE> *vkb_save_ptr, 
    int nkb, 
    int gcar_zero_count,
    int npw,  
    int ipol,
    int npwx,
    const std::complex<FPTYPE> coeff)
{
    hipLaunchKernelGGL(HIP_KERNEL_NAME(revertVkbValues_<FPTYPE>), dim3(nkb*gcar_zero_count), dim3(THREADS_PER_BLOCK), 0, 0,
        gcar_zero_ptrs, 
        reinterpret_cast<thrust::complex<FPTYPE>*>(vkb_ptr), 
        reinterpret_cast<const thrust::complex<FPTYPE>*>(vkb_save_ptr), 
        nkb, 
        npw, 
        ipol,
        npwx,
        static_cast<const thrust::complex<FPTYPE>>(coeff));
    hipCheckOnDebug();
}

// Local-pseudopotential and Ewald forces, ported from ../cuda/force_op.cu. One block per
// atom; the CUDA warp-shuffle reduction (32-lane warps) is replaced by a shared-memory tree
// reduction, which does not depend on the wavefront width.
template <typename FPTYPE>
__device__ void block_reduce3(FPTYPE& x, FPTYPE& y, FPTYPE& z)
{
    __shared__ FPTYPE sx[THREADS_PER_BLOCK];
    __shared__ FPTYPE sy[THREADS_PER_BLOCK];
    __shared__ FPTYPE sz[THREADS_PER_BLOCK];
    const int tid = threadIdx.x;
    sx[tid] = x; sy[tid] = y; sz[tid] = z;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sx[tid] += sx[tid + stride];
            sy[tid] += sy[tid + stride];
            sz[tid] += sz[tid + stride];
        }
        __syncthreads();
    }
    x = sx[0]; y = sy[0]; z = sz[0];
}

template <typename FPTYPE>
__global__ void force_loc_kernel(
    const int nat,
    const int npw,
    const FPTYPE tpiba_omega,
    const int* iat2it,
    const int* ig2gg_d,
    const FPTYPE* gcar_d,
    const FPTYPE* tau_d,
    const thrust::complex<FPTYPE>* aux_d,
    const FPTYPE* vloc_d,
    const int vloc_nc,
    FPTYPE* forcelc_d)
{
    const FPTYPE two_pi = 2.0 * 3.14159265358979323846;
    const int iat = blockIdx.x;
    if (iat >= nat) return;
    const int it = iat2it[iat];
    FPTYPE force_x = 0.0, force_y = 0.0, force_z = 0.0;
    const FPTYPE tau_x = tau_d[iat * 3 + 0];
    const FPTYPE tau_y = tau_d[iat * 3 + 1];
    const FPTYPE tau_z = tau_d[iat * 3 + 2];
    for (int ig = threadIdx.x; ig < npw; ig += blockDim.x)
    {
        const FPTYPE gcar_x = gcar_d[ig * 3 + 0];
        const FPTYPE gcar_y = gcar_d[ig * 3 + 1];
        const FPTYPE gcar_z = gcar_d[ig * 3 + 2];
        const FPTYPE phase = two_pi * (gcar_x * tau_x + gcar_y * tau_y + gcar_z * tau_z);
        const FPTYPE sinp = sin(phase);
        const FPTYPE cosp = cos(phase);
        const FPTYPE vloc_val = vloc_d[it * vloc_nc + ig2gg_d[ig]];
        const thrust::complex<FPTYPE> aux_val = aux_d[ig];
        const FPTYPE factor = vloc_val * (cosp * aux_val.imag() + sinp * aux_val.real());
        force_x += gcar_x * factor;
        force_y += gcar_y * factor;
        force_z += gcar_z * factor;
    }
    block_reduce3(force_x, force_y, force_z);
    if (threadIdx.x == 0)
    {
        forcelc_d[iat * 3 + 0] = force_x * tpiba_omega;
        forcelc_d[iat * 3 + 1] = force_y * tpiba_omega;
        forcelc_d[iat * 3 + 2] = force_z * tpiba_omega;
    }
}

template <typename FPTYPE>
__global__ void force_ew_kernel(
    const int nat,
    const int npw,
    const int ig_gge0,
    const int* iat2it,
    const FPTYPE* gcar_d,
    const FPTYPE* tau_d,
    const FPTYPE* it_fact_d,
    const thrust::complex<FPTYPE>* aux_d,
    FPTYPE* forceion_d)
{
    const FPTYPE two_pi = 2.0 * 3.14159265358979323846;
    const int iat = blockIdx.x;
    if (iat >= nat) return;
    const int it = iat2it[iat];
    const FPTYPE it_fact_val = it_fact_d[it];
    FPTYPE force_x = 0.0, force_y = 0.0, force_z = 0.0;
    const FPTYPE tau_x = tau_d[iat * 3 + 0];
    const FPTYPE tau_y = tau_d[iat * 3 + 1];
    const FPTYPE tau_z = tau_d[iat * 3 + 2];
    for (int ig = threadIdx.x; ig < npw; ig += blockDim.x)
    {
        if (ig == ig_gge0) continue;
        const FPTYPE gcar_x = gcar_d[ig * 3 + 0];
        const FPTYPE gcar_y = gcar_d[ig * 3 + 1];
        const FPTYPE gcar_z = gcar_d[ig * 3 + 2];
        const FPTYPE phase = two_pi * (gcar_x * tau_x + gcar_y * tau_y + gcar_z * tau_z);
        const FPTYPE sinp = sin(phase);
        const FPTYPE cosp = cos(phase);
        const FPTYPE sumnb = -cosp * aux_d[ig].imag() + sinp * aux_d[ig].real();
        force_x += gcar_x * sumnb;
        force_y += gcar_y * sumnb;
        force_z += gcar_z * sumnb;
    }
    block_reduce3(force_x, force_y, force_z);
    if (threadIdx.x == 0)
    {
        forceion_d[iat * 3 + 0] = force_x * it_fact_val;
        forceion_d[iat * 3 + 1] = force_y * it_fact_val;
        forceion_d[iat * 3 + 2] = force_z * it_fact_val;
    }
}

template <typename FPTYPE>
void cal_force_loc_op<FPTYPE, base_device::DEVICE_GPU>::operator()(
    const int nat,
    const int npw,
    const FPTYPE tpiba_omega,
    const int* iat2it,
    const int* ig2igg,
    const FPTYPE* gcar,
    const FPTYPE* tau,
    const std::complex<FPTYPE>* aux,
    const FPTYPE* vloc,
    const int vloc_nc,
    FPTYPE* forcelc)
{
    hipLaunchKernelGGL(HIP_KERNEL_NAME(force_loc_kernel<FPTYPE>), dim3(nat), dim3(THREADS_PER_BLOCK), 0, 0,
        nat, npw, tpiba_omega, iat2it, ig2igg, gcar, tau,
        reinterpret_cast<const thrust::complex<FPTYPE>*>(aux), vloc, vloc_nc, forcelc);
    hipCheckOnDebug();
}

template <typename FPTYPE>
void cal_force_ew_op<FPTYPE, base_device::DEVICE_GPU>::operator()(
    const int nat,
    const int npw,
    const int ig_gge0,
    const int* iat2it,
    const FPTYPE* gcar,
    const FPTYPE* tau,
    const FPTYPE* it_fact,
    const std::complex<FPTYPE>* aux,
    FPTYPE* forceion)
{
    hipLaunchKernelGGL(HIP_KERNEL_NAME(force_ew_kernel<FPTYPE>), dim3(nat), dim3(THREADS_PER_BLOCK), 0, 0,
        nat, npw, ig_gge0, iat2it, gcar, tau, it_fact,
        reinterpret_cast<const thrust::complex<FPTYPE>*>(aux), forceion);
    hipCheckOnDebug();
}

// for revertVkbValues functions instantiation
template void revertVkbValues<double>(const int *gcar_zero_ptrs, std::complex<double> *vkb_ptr, const std::complex<double> *vkb_save_ptr, int nkb, int gcar_zero_count, int npw, int ipol, int npwx, const std::complex<double> coeff);
template void revertVkbValues<float>(const int *gcar_zero_ptrs, std::complex<float> *vkb_ptr, const std::complex<float> *vkb_save_ptr, int nkb, int gcar_zero_count, int npw, int ipol, int npwx, const std::complex<float> coeff);
// for saveVkbValues functions instantiation
template void saveVkbValues<double>(const int *gcar_zero_ptrs, const std::complex<double> *vkb_ptr, std::complex<double> *vkb_save_ptr, int nkb, int gcar_zero_count, int npw, int ipol, int npwx);
template void saveVkbValues<float>(const int *gcar_zero_ptrs, const std::complex<float> *vkb_ptr, std::complex<float> *vkb_save_ptr, int nkb, int gcar_zero_count, int npw, int ipol, int npwx);

template struct cal_vkb1_nl_op<float, base_device::DEVICE_GPU>;
template struct cal_force_nl_op<float, base_device::DEVICE_GPU>;
template struct cal_force_loc_op<float, base_device::DEVICE_GPU>;
template struct cal_force_ew_op<float, base_device::DEVICE_GPU>;

template struct cal_vkb1_nl_op<double, base_device::DEVICE_GPU>;
template struct cal_force_nl_op<double, base_device::DEVICE_GPU>;
template struct cal_force_loc_op<double, base_device::DEVICE_GPU>;
template struct cal_force_ew_op<double, base_device::DEVICE_GPU>;

}  // namespace hamilt
