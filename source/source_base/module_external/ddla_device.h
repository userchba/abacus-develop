#ifndef ABACUS_SOURCE_BASE_MODULE_EXTERNAL_DDLA_DEVICE_H
#define ABACUS_SOURCE_BASE_MODULE_EXTERNAL_DDLA_DEVICE_H

// Block-cyclic matrices in the memory of a LibDDLA handle: device memory for a
// GPU handle, host memory for a CPU handle. Include only from source files
// that are compiled with LibDDLA (__DDLA).
#ifdef __DDLA
#include "source_base/module_external/ddla_context.h"
#include "source_base/module_external/blacs_connector.h"
#include "source_base/module_external/scalapack_connector.h"

#include <ddla/ddla.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <type_traits>

namespace ModuleBase
{
namespace ddla_device
{

// Storage for the local block of a matrix of scalar type T in the handle's
// memory; the default T is the complex propagation type.
template <typename T = std::complex<double>>
using DeviceMatrixOf = ddla::vector<T>;
using DeviceMatrix = DeviceMatrixOf<>;

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

// Whether the local block of desc on this rank has rows beyond the matrix:
// its leading dimension exceeds its row count, as for a narrow matrix that
// shares the leading dimension of a larger layout (desc_wfc, desc_Eij).
inline bool has_padding_rows(const int* desc)
{
    int nprow = 0;
    int npcol = 0;
    int myrow = 0;
    int mycol = 0;
    Cblacs_gridinfo(desc[1], &nprow, &npcol, &myrow, &mycol);
    const int nrow = numroc_(&desc[2], &desc[4], &myrow, &desc[6], &nprow);
    return local_size(desc) > 0 && nrow < desc[8];
}

// Copy the local block of a host matrix into an existing matrix in the handle's memory.
template <typename T>
inline void copy_to_device(const ddla::DdlaHandle_t& handle,
                           DeviceMatrixOf<T>& device,
                           const T* host,
                           const int* desc)
{
    const std::size_t bytes = local_size(desc) * sizeof(T);
    ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle, device.data(), host, bytes));
}

// Matrix for the layout desc in the handle's memory, with uninitialized
// content. Ranks without local data keep one element so that LibDDLA always
// gets a valid pointer.
template <typename T = std::complex<double>>
inline DeviceMatrixOf<T> allocate(const ddla::DdlaHandle_t& handle, const int* desc)
{
    const std::size_t count = local_size(desc);
    const std::size_t capacity = std::max<std::size_t>(count, 1);
    return DeviceMatrixOf<T>(handle, capacity);
}

// Matrix for the layout desc in the handle's memory, filled with zeros.
template <typename T = std::complex<double>>
inline DeviceMatrixOf<T> zeros(const ddla::DdlaHandle_t& handle, const int* desc)
{
    DeviceMatrixOf<T> device = allocate<T>(handle, desc);
    const std::size_t bytes = device.size() * sizeof(T);
    ddla::DDLA_CHECK(ddla::ddlaMemset(handle, device.data(), 0, bytes));
    return device;
}

// Copy of the local block of a host matrix in the handle's memory.
template <typename T>
inline DeviceMatrixOf<T> to_device(const ddla::DdlaHandle_t& handle, const T* host, const int* desc)
{
    DeviceMatrixOf<T> device = allocate<T>(handle, desc);
    copy_to_device(handle, device, host, desc);
    return device;
}

// Copy the local block of a matrix in the handle's memory back to host memory and wait for it.
template <typename T>
inline void to_host(const ddla::DdlaHandle_t& handle,
                    T* host,
                    const DeviceMatrixOf<T>& device,
                    const int* desc)
{
    const std::size_t bytes = local_size(desc) * sizeof(T);
    ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle, host, device.data(), bytes));
    ddla::DDLA_CHECK(ddla::ddlaSynchronize(handle));
}

// The local block of a host matrix where the backend of a LibDDLA call reads
// and writes it: the host storage itself for a CPU handle, a device copy for
// a GPU handle. T may be const for an input that is only read.
template <typename T>
class BackendBlock
{
  public:
    using Value = typename std::remove_const<T>::type;

    BackendBlock(const DdlaContext& ddla, T* host, const int* desc) : BackendBlock(ddla, host, desc, true)
    {
    }

    // copy_in false: the block is only written, so a GPU handle allocates
    // it on the GPU without copying the host content there.
    BackendBlock(const DdlaContext& ddla, T* host, const int* desc, const bool copy_in)
        : handle_(ddla.handle()), on_gpu_(ddla.on_gpu()), host_(host), desc_(desc), device_(ddla.handle())
    {
        if (on_gpu_)
        {
            device_ = copy_in ? to_device<Value>(handle_, host_, desc_) : allocate<Value>(handle_, desc_);
        }
    }

    // Pointer to pass to LibDDLA.
    T* data()
    {
        return on_gpu_ ? device_.data() : host_;
    }

    // Make a result visible in the host matrix: copy it back from the GPU;
    // a CPU handle has already written it there.
    void to_host()
    {
        if (on_gpu_)
        {
            ddla_device::to_host(handle_, host_, device_, desc_);
        }
    }

  private:
    const ddla::DdlaHandle_t& handle_;
    const bool on_gpu_;
    T* host_;
    const int* desc_;
    DeviceMatrixOf<Value> device_;
};

} // namespace ddla_device
} // namespace ModuleBase

#endif // __DDLA

#endif // ABACUS_SOURCE_BASE_MODULE_EXTERNAL_DDLA_DEVICE_H
