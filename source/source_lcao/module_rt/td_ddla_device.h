#ifndef ABACUS_SOURCE_LCAO_MODULE_RT_TD_DDLA_DEVICE_H
#define ABACUS_SOURCE_LCAO_MODULE_RT_TD_DDLA_DEVICE_H

// GPU staging of block-cyclic matrices for LibDDLA. Include only from source
// files that are compiled with LibDDLA (__DDLA); the content is available
// when LibDDLA has GPU support.
#ifdef __DDLA
#include <ddla/ddla.h>

#if DDLA_HAS_GPU
#include "source_base/module_external/blacs_connector.h"
#include "source_base/module_external/scalapack_connector.h"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace module_rt
{
namespace ddla_device
{

using DeviceMatrix = ddla::vector<std::complex<double>>;

// Number of locally stored elements (leading dimension x local columns) of
// the block-cyclic matrix described by desc on this rank; zero when the rank
// owns no row or no column, where the leading dimension is only a placeholder.
inline std::size_t local_size(const int* desc)
{
    int nprow = 0;
    int npcol = 0;
    int myrow = 0;
    int mycol = 0;
    Cblacs_gridinfo(desc[1], &nprow, &npcol, &myrow, &mycol);
    const int nrow = numroc_(&desc[2], &desc[4], &myrow, &desc[6], &nprow);
    const int ncol = numroc_(&desc[3], &desc[5], &mycol, &desc[7], &npcol);
    if (nrow <= 0 || ncol <= 0)
    {
        return 0;
    }
    return static_cast<std::size_t>(desc[8]) * static_cast<std::size_t>(ncol);
}

// Copy the local block of a host matrix into an existing device matrix.
inline void copy_to_device(const ddla::DdlaHandle_t& handle,
                           DeviceMatrix& device,
                           const std::complex<double>* host,
                           const int* desc)
{
    const std::size_t count = local_size(desc);
    const std::size_t bytes = count * sizeof(std::complex<double>);
    const ddla::ddlaStatus_t status = ddla::ddlaMemcpy(handle, device.data(), host, bytes);
    if (status != ddla::ddlaStatus_t::DDLA_STATUS_SUCCESS)
    {
        const std::string message = "host-to-device copy of " + std::to_string(count) + " elements failed with status "
                                    + std::to_string(static_cast<int>(status));
        throw std::runtime_error(message);
    }
}

// Device matrix for the layout desc with uninitialized content. Ranks without
// local data keep one element so that LibDDLA always gets a valid pointer.
inline DeviceMatrix allocate(const ddla::DdlaHandle_t& handle, const int* desc)
{
    const std::size_t count = local_size(desc);
    const std::size_t capacity = std::max<std::size_t>(count, 1);
    return DeviceMatrix(handle, capacity);
}

// Device matrix for the layout desc filled with zeros.
inline DeviceMatrix zeros(const ddla::DdlaHandle_t& handle, const int* desc)
{
    DeviceMatrix device = allocate(handle, desc);
    const std::size_t bytes = device.size() * sizeof(std::complex<double>);
    const ddla::ddlaStatus_t status = ddla::ddlaMemset(handle, device.data(), 0, bytes);
    if (status != ddla::ddlaStatus_t::DDLA_STATUS_SUCCESS)
    {
        const std::string message = "device memset failed with status " + std::to_string(static_cast<int>(status));
        throw std::runtime_error(message);
    }
    return device;
}

// Copy the local block of a host matrix to the handle's GPU.
inline DeviceMatrix to_device(const ddla::DdlaHandle_t& handle, const std::complex<double>* host, const int* desc)
{
    DeviceMatrix device = allocate(handle, desc);
    copy_to_device(handle, device, host, desc);
    return device;
}

// Copy the local block of a GPU matrix back to host memory and wait for it.
inline void to_host(const ddla::DdlaHandle_t& handle,
                    std::complex<double>* host,
                    const DeviceMatrix& device,
                    const int* desc)
{
    const std::size_t count = local_size(desc);
    const std::size_t bytes = count * sizeof(std::complex<double>);
    const ddla::ddlaStatus_t copy_status = ddla::ddlaMemcpy(handle, host, device.data(), bytes);
    const ddla::ddlaStatus_t sync_status = ddla::ddlaSynchronize(handle);
    if (copy_status != ddla::ddlaStatus_t::DDLA_STATUS_SUCCESS || sync_status != ddla::ddlaStatus_t::DDLA_STATUS_SUCCESS)
    {
        const std::string message = "device-to-host copy of " + std::to_string(count) + " elements failed with status "
                                    + std::to_string(static_cast<int>(copy_status)) + "/"
                                    + std::to_string(static_cast<int>(sync_status));
        throw std::runtime_error(message);
    }
}

} // namespace ddla_device
} // namespace module_rt

#endif // DDLA_HAS_GPU
#endif // __DDLA

#endif // ABACUS_SOURCE_LCAO_MODULE_RT_TD_DDLA_DEVICE_H
