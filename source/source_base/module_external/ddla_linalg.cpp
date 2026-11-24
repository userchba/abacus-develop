#include "source_base/module_external/ddla_linalg.h"

#include "source_base/module_external/blacs_connector.h"
#include "source_base/module_external/scalapack_connector.h"
#include "source_base/tool_quit.h"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <vector>
#include <stdexcept>
#include <utility>

#ifdef __DDLA
#include "source_base/block_cyclic_distribution.h"
#include "source_base/module_external/ddla_context.h"
#include "source_base/module_external/ddla_device.h"

#include <ddla/ddla.h>

#include <exception>
#include <string>
#endif

namespace ModuleBase
{
namespace ddla_linalg
{
const DdlaContext* gpu_backend(const DdlaContext* ddla)
{
#ifdef __DDLA
    if (ddla != nullptr && ddla->on_gpu())
    {
        return ddla;
    }
#else
    (void)ddla;
#endif
    return nullptr;
}

const DdlaContext* bound_backend(const DdlaContext* ddla, const BlockCyclicDistribution& dist)
{
#ifdef __DDLA
    if (ddla != nullptr && ddla->is_bound() && ddla->ictxt() == dist.ictxt())
    {
        return ddla;
    }
#else
    (void)ddla;
    (void)dist;
#endif
    return nullptr;
}

#ifdef __MPI
namespace
{
#ifdef __DDLA
template <typename T>
using Block = ddla_device::BackendBlock<T>;

template <typename T>
using ConstBlock = ddla_device::BackendBlock<const T>;

void quit_on_ddla_error(const char* where, const std::exception& e)
{
    const std::string message = std::string("LibDDLA call failed: ") + e.what();
    ModuleBase::WARNING_QUIT(where, message);
}
#endif // __DDLA

void quit_without_ddla(const char* where)
{
    ModuleBase::WARNING_QUIT(where, "ABACUS is built without LibDDLA support!");
}

// Number of rows of the matrix described by desc that this rank stores.
int local_rows(const int* desc)
{
    int nprow = 0;
    int npcol = 0;
    int myrow = 0;
    int mycol = 0;
    Cblacs_gridinfo(desc[1], &nprow, &npcol, &myrow, &mycol);
    return numroc_(&desc[2], &desc[4], &myrow, &desc[6], &nprow);
}

#ifdef __DDLA
// Descriptor of a whole rows x cols matrix with the blocks and the BLACS grid
// of desc, its first block on process (0, 0).
void whole_desc(const int* desc, const int rows, const int cols, int* out)
{
    int nprow = 0;
    int npcol = 0;
    int myrow = 0;
    int mycol = 0;
    Cblacs_gridinfo(desc[1], &nprow, &npcol, &myrow, &mycol);
    const int zero = 0;
    const int nrloc = numroc_(&rows, &desc[4], &myrow, &zero, &nprow);
    const int whole[9] = {1, desc[1], rows, cols, desc[4], desc[5], 0, 0, std::max(nrloc, 1)};
    std::copy(whole, whole + 9, out);
}

// Copy of the rows x cols submatrix of A starting at (ia, ja), as a whole
// matrix with descriptor desc_out in the handle's memory; A is in the
// handle's memory too.
template <typename T>
ddla_device::DeviceMatrixOf<T> extract(const DdlaContext& ddla,
                                       const T* A,
                                       const int ia,
                                       const int ja,
                                       const int* descA,
                                       const int rows,
                                       const int cols,
                                       int* desc_out)
{
    whole_desc(descA, rows, cols, desc_out);
    ddla_device::DeviceMatrixOf<T> sub = ddla_device::allocate<T>(ddla.handle(), desc_out);
    const char no_trans = 'N';
    const T one(1.0);
    const T zero(0.0);
    ddla::pgeadd(ddla.handle(),
                 no_trans,
                 no_trans,
                 rows,
                 cols,
                 one,
                 A,
                 ia,
                 ja,
                 descA,
                 zero,
                 sub.data(),
                 1,
                 1,
                 desc_out,
                 sub.data(),
                 1,
                 1,
                 desc_out);
    return sub;
}
#endif // __DDLA
} // namespace


template <typename T>
struct ResidentMatrix<T>::Impl
{
    const DdlaContext* context;
    int descriptor[9];
    std::size_t count;
    const T* readable = nullptr;
    T* writable = nullptr;
    bool can_write;
    bool gpu = false;
    std::vector<T> host_storage;
#ifdef __DDLA
    std::unique_ptr<ddla_device::DeviceMatrixOf<T>> device;
#endif

    Impl(const DdlaContext* ctx, const T* input, T* output, const int* desc, bool copy_in, bool own, bool writeable)
        : context(ctx), count(0), can_write(writeable)
    {
        std::copy(desc, desc + 9, descriptor);
        int nprow = 0;
        int npcol = 0;
        int myrow = 0;
        int mycol = 0;
        Cblacs_gridinfo(desc[1], &nprow, &npcol, &myrow, &mycol);
        const int rows = numroc_(&desc[2], &desc[4], &myrow, &desc[6], &nprow);
        const int cols = numroc_(&desc[3], &desc[5], &mycol, &desc[7], &npcol);
        if (rows > 0 && cols > 0)
        {
            count = static_cast<std::size_t>(desc[8]) * static_cast<std::size_t>(cols);
        }
        if (writeable && !own && count > 0 && output == nullptr)
        {
            throw std::invalid_argument("Resident matrix requires a host output for its nonempty local block");
        }
        const bool preserve_padding = output != nullptr && rows < desc[8];
        const bool read_input = copy_in || preserve_padding;
        if (read_input && count > 0 && input == nullptr)
        {
            throw std::invalid_argument("Resident matrix requires a host input for its nonempty local block");
        }
#ifdef __DDLA
        if (ctx != nullptr && (!ctx->is_bound() || ctx->ictxt() != desc[1]))
        {
            throw std::invalid_argument("Resident matrix and LibDDLA context must use the same bound process grid");
        }
        gpu = ctx != nullptr && ctx->on_gpu();
        if (gpu)
        {
            device.reset(new ddla_device::DeviceMatrixOf<T>(ctx->handle()));
            if (own)
            {
                *device = ddla_device::zeros<T>(ctx->handle(), descriptor);
            }
            else
            {
                *device = ddla_device::allocate<T>(ctx->handle(), descriptor);
            }
            if (read_input && count > 0)
            {
                ddla_device::copy_to_device(ctx->handle(), *device, input, descriptor);
            }
            readable = device->data();
            if (can_write)
            {
                writable = device->data();
            }
            return;
        }
#else
        if (ctx != nullptr)
        {
            quit_without_ddla("ResidentMatrix");
        }
#endif
        if (own || count == 0)
        {
            const std::size_t capacity = std::max<std::size_t>(count, 1);
            host_storage.assign(capacity, T(0.0));
            readable = host_storage.data();
            if (can_write || count == 0)
            {
                writable = host_storage.data();
            }
        }
        else
        {
            readable = input;
            writable = output;
        }
    }
};

template <typename T>
ResidentMatrix<T>::ResidentMatrix(const DdlaContext* context, const T* input, const int* desc)
{
    impl_.reset(new Impl(context, input, nullptr, desc, true, false, false));
}
template <typename T>
ResidentMatrix<T>::ResidentMatrix(const DdlaContext* context, T* output, const int* desc, bool copy_in)
{
    impl_.reset(new Impl(context, output, output, desc, copy_in, false, true));
}
template <typename T>
ResidentMatrix<T>::ResidentMatrix(const DdlaContext* context, const int* desc)
{
    impl_.reset(new Impl(context, nullptr, nullptr, desc, false, true, true));
}
template <typename T> ResidentMatrix<T>::~ResidentMatrix() = default;
template <typename T> ResidentMatrix<T>::ResidentMatrix(ResidentMatrix&&) noexcept = default;
template <typename T> ResidentMatrix<T>& ResidentMatrix<T>::operator=(ResidentMatrix&&) noexcept = default;
template <typename T> const DdlaContext* ResidentMatrix<T>::context() const { return impl_->context; }
template <typename T> const int* ResidentMatrix<T>::desc() const { return impl_->descriptor; }
template <typename T> const T* ResidentMatrix<T>::data() const { return impl_->readable; }
template <typename T> T* ResidentMatrix<T>::data()
{
    if (!impl_->can_write)
    {
        throw std::logic_error("Cannot write a read-only resident matrix");
    }
    return impl_->writable;
}
template <typename T>
void ResidentMatrix<T>::refresh(const T* input)
{
    if (impl_->count == 0)
    {
        return;
    }
    if (input == nullptr)
    {
        throw std::invalid_argument("Cannot refresh a nonempty matrix from a null host input");
    }
#ifdef __DDLA
    if (impl_->gpu)
    {
        if (impl_->count > 0)
        {
            ddla_device::copy_to_device(impl_->context->handle(), *impl_->device, input, impl_->descriptor);
        }
        return;
    }
#endif
    if (impl_->can_write)
    {
        std::copy(input, input + impl_->count, impl_->writable);
    }
    else
    {
        impl_->readable = input;
    }
}
template <typename T>
void ResidentMatrix<T>::copy_to_host(T* output) const
{
    if (impl_->count > 0 && output == nullptr)
    {
        throw std::invalid_argument("Cannot copy a nonempty matrix to a null host output");
    }
#ifdef __DDLA
    if (impl_->gpu)
    {
        if (impl_->count > 0)
        {
            ddla_device::to_host(impl_->context->handle(), output, *impl_->device, impl_->descriptor);
        }
        else
        {
            ddla::DDLA_CHECK(ddla::ddlaSynchronize(impl_->context->handle()));
        }
        return;
    }
#endif
    if (impl_->count > 0 && output != impl_->readable)
    {
        std::copy(impl_->readable, impl_->readable + impl_->count, output);
    }
}

template <typename T>
void gemm_impl(const DdlaContext* ddla,
          const char transa,
          const char transb,
          const int m,
          const int n,
          const int k,
          const typename Scalar<T>::type& alpha,
          const T* A,
          const int* descA,
          const T* B,
          const int* descB,
          const typename Scalar<T>::type& beta,
          T* C,
          const int* descC,
          const bool stage_host)
{
    if (ddla == nullptr)
    {
        ScalapackConnector::gemm(transa, transb, m, n, k, alpha, A, 1, 1, descA, B, 1, 1, descB, beta, C, 1, 1, descC);
        return;
    }
#ifdef __DDLA
    try
    {
        ConstBlock<T> a(*ddla, A, descA, true, stage_host);
        ConstBlock<T> b(*ddla, B, descB, true, stage_host);
        // With beta = 0 the old content of C is not read, so a GPU handle
        // does not copy it to the GPU, unless the local block has padding
        // rows: copying the block back would then overwrite them.
        const bool read_c = !(beta == T(0.0)) || ddla_device::has_padding_rows(descC);
        Block<T> c(*ddla, C, descC, read_c, stage_host);
        ddla::pgemm(ddla->handle(), transa, transb, m, n, k, alpha, a.data(), descA, b.data(), descB, beta, c.data(), descC);
        c.to_host();
    }
    catch (const std::exception& e)
    {
        quit_on_ddla_error("ModuleBase::ddla_linalg::gemm", e);
    }
#else
    quit_without_ddla("ModuleBase::ddla_linalg::gemm");
#endif // __DDLA
}

template <typename T>
void gemm_impl(const DdlaContext* ddla,
          const char transa,
          const char transb,
          const int m,
          const int n,
          const int k,
          const typename Scalar<T>::type& alpha,
          const T* A,
          const int ia,
          const int ja,
          const int* descA,
          const T* B,
          const int ib,
          const int jb,
          const int* descB,
          const typename Scalar<T>::type& beta,
          T* C,
          const int ic,
          const int jc,
          const int* descC,
          const bool stage_host)
{
    if (ddla == nullptr)
    {
        ScalapackConnector::gemm(transa, transb, m, n, k, alpha, A, ia, ja, descA, B, ib, jb, descB, beta, C, ic, jc,
                                 descC);
        return;
    }
#ifdef __DDLA
    if (m == 0 || n == 0)
    {
        return;
    }
    try
    {
        const ddla::DdlaHandle_t& handle = ddla->handle();
        ConstBlock<T> a(*ddla, A, descA, true, stage_host);
        ConstBlock<T> b(*ddla, B, descB, true, stage_host);
        // Operands with an offset are copied into whole matrices of their
        // stored shape (before op() is applied).
        const T* a_data = a.data();
        const int* a_desc = descA;
        int a_sub_desc[9];
        ddla_device::DeviceMatrixOf<T> a_sub(handle);
        if ((ia != 1 || ja != 1) && k > 0)
        {
            const int rows = transa == 'N' ? m : k;
            const int cols = transa == 'N' ? k : m;
            a_sub = extract(*ddla, a.data(), ia, ja, descA, rows, cols, a_sub_desc);
            a_data = a_sub.data();
            a_desc = a_sub_desc;
        }
        const T* b_data = b.data();
        const int* b_desc = descB;
        int b_sub_desc[9];
        ddla_device::DeviceMatrixOf<T> b_sub(handle);
        if ((ib != 1 || jb != 1) && k > 0)
        {
            const int rows = transb == 'N' ? k : n;
            const int cols = transb == 'N' ? n : k;
            b_sub = extract(*ddla, b.data(), ib, jb, descB, rows, cols, b_sub_desc);
            b_data = b_sub.data();
            b_desc = b_sub_desc;
        }
        // C is read when beta != 0, and copied in whole when only a part of it
        // is written or its local block has padding rows (copying the block
        // back would overwrite them).
        const bool read_c = !(beta == T(0.0));
        const bool c_offset = ic != 1 || jc != 1;
        Block<T> c(*ddla, C, descC, read_c || c_offset || ddla_device::has_padding_rows(descC), stage_host);
        T* c_data = c.data();
        const int* c_desc = descC;
        int c_sub_desc[9];
        ddla_device::DeviceMatrixOf<T> c_sub(handle);
        if (c_offset)
        {
            if (read_c)
            {
                c_sub = extract(*ddla, static_cast<const T*>(c.data()), ic, jc, descC, m, n, c_sub_desc);
            }
            else
            {
                whole_desc(descC, m, n, c_sub_desc);
                c_sub = ddla_device::allocate<T>(handle, c_sub_desc);
            }
            c_data = c_sub.data();
            c_desc = c_sub_desc;
        }
        ddla::pgemm(handle, transa, transb, m, n, k, alpha, a_data, a_desc, b_data, b_desc, beta, c_data, c_desc);
        if (c_offset)
        {
            const char no_trans = 'N';
            const T one(1.0);
            const T zero(0.0);
            ddla::pgeadd(handle,
                         no_trans,
                         no_trans,
                         m,
                         n,
                         one,
                         c_sub.data(),
                         1,
                         1,
                         c_sub_desc,
                         zero,
                         c_sub.data(),
                         1,
                         1,
                         c_sub_desc,
                         c.data(),
                         ic,
                         jc,
                         descC);
        }
        c.to_host();
    }
    catch (const std::exception& e)
    {
        quit_on_ddla_error("ModuleBase::ddla_linalg::gemm", e);
    }
#else
    quit_without_ddla("ModuleBase::ddla_linalg::gemm");
#endif // __DDLA
}

template <typename T>
void tran(const DdlaContext* ddla,
          const char trans,
          const int m,
          const int n,
          const typename Scalar<T>::type& alpha,
          const T* A,
          const int* descA,
          const typename Scalar<T>::type& beta,
          T* C,
          const int* descC)
{
    if (trans != 'T' && trans != 'C')
    {
        ModuleBase::WARNING_QUIT("ModuleBase::ddla_linalg::tran", "trans must be 'T' or 'C'");
    }
    if (ddla == nullptr)
    {
        if (trans == 'T')
        {
            ScalapackConnector::tranu(m, n, alpha, A, 1, 1, descA, beta, C, 1, 1, descC);
        }
        else
        {
            ScalapackConnector::tranc(m, n, alpha, A, 1, 1, descA, beta, C, 1, 1, descC);
        }
        return;
    }
    geadd(ddla, trans, m, n, alpha, A, descA, beta, C, descC);
}

template <typename T>
int gesv_impl(const DdlaContext* ddla, const int n, const int nrhs, T* A, const int* descA, T* B, const int* descB,
          const bool stage_host)
{
    int info = 0;
    // PxGESV and LibDDLA pgetrf need LOCr(M_A) + MB_A pivot entries.
    std::vector<int> ipiv(static_cast<std::size_t>(local_rows(descA)) + static_cast<std::size_t>(descA[4]), 0);
    if (ddla == nullptr)
    {
        ScalapackConnector::gesv(n, nrhs, A, 1, 1, descA, ipiv.data(), B, 1, 1, descB, &info);
        return info;
    }
#ifdef __DDLA
    try
    {
        const ddla::DdlaHandle_t& handle = ddla->handle();
        Block<T> a(*ddla, A, descA, true, stage_host);
        Block<T> b(*ddla, B, descB, true, stage_host);
        ddla::vector<int> pivots(handle, ipiv.size());
        ddla::pgetrf(handle, n, n, a.data(), 1, 1, descA, pivots.data(), info);
        if (info == 0)
        {
            const char left = 'L';
            const char no_trans = 'N';
            ddla::pgetrs(handle, left, no_trans, n, nrhs, a.data(), descA, pivots.data(), b.data(), descB);
            b.to_host();
        }
        a.to_host();
    }
    catch (const std::exception& e)
    {
        quit_on_ddla_error("ModuleBase::ddla_linalg::gesv", e);
    }
#else
    quit_without_ddla("ModuleBase::ddla_linalg::gesv");
#endif // __DDLA
    return info;
}

template <typename T>
int posv(const DdlaContext* ddla,
         const char uplo,
         const int n,
         const int nrhs,
         T* A,
         const int* descA,
         T* B,
         const int* descB)
{
    int info = 0;
    if (ddla == nullptr)
    {
        int desc_a[9];
        std::copy(descA, descA + 9, desc_a);
        info = ScalapackConnector::potrf(uplo, n, A, desc_a);
        if (info == 0)
        {
            ScalapackConnector::potrs(uplo, n, nrhs, A, 1, 1, descA, B, 1, 1, descB, &info);
        }
        return info;
    }
#ifdef __DDLA
    try
    {
        const ddla::DdlaHandle_t& handle = ddla->handle();
        Block<T> a(*ddla, A, descA);
        Block<T> b(*ddla, B, descB);
        ddla::ppotrf(handle, uplo, n, a.data(), 1, 1, descA, info);
        if (info == 0)
        {
            const char left = 'L';
            const char no_trans = 'N';
            const T* factor = a.data();
            ddla::ppotrs(handle, left, uplo, no_trans, n, nrhs, factor, descA, b.data(), descB);
            b.to_host();
        }
        a.to_host();
    }
    catch (const std::exception& e)
    {
        quit_on_ddla_error("ModuleBase::ddla_linalg::posv", e);
    }
#else
    quit_without_ddla("ModuleBase::ddla_linalg::posv");
#endif // __DDLA
    return info;
}

template <typename T>
void geadd_impl(const DdlaContext* ddla,
           const char trans,
           const int m,
           const int n,
           const typename Scalar<T>::type& alpha,
           const T* A,
           const int* descA,
           const typename Scalar<T>::type& beta,
           T* C,
           const int* descC,
          const bool stage_host)
{
    if (ddla == nullptr)
    {
        ScalapackConnector::geadd(trans, m, n, alpha, A, 1, 1, descA, beta, C, 1, 1, descC);
        return;
    }
#ifdef __DDLA
    // LibDDLA computes C = alpha * op(A) + beta * op(B); B = C reproduces PxGEADD.
    const char no_trans = 'N';
    try
    {
        ConstBlock<T> a(*ddla, A, descA, true, stage_host);
        Block<T> c(*ddla, C, descC, true, stage_host);
        ddla::pgeadd(ddla->handle(),
                     trans,
                     no_trans,
                     m,
                     n,
                     alpha,
                     a.data(),
                     1,
                     1,
                     descA,
                     beta,
                     c.data(),
                     1,
                     1,
                     descC,
                     c.data(),
                     1,
                     1,
                     descC);
        c.to_host();
    }
    catch (const std::exception& e)
    {
        quit_on_ddla_error("ModuleBase::ddla_linalg::geadd", e);
    }
#else
    quit_without_ddla("ModuleBase::ddla_linalg::geadd");
#endif // __DDLA
}

template <typename T>
void getrf(const DdlaContext& ddla, const BlockCyclicDistribution& dist, T* A, std::vector<int>& ipiv)
{
#ifdef __DDLA
    const int n = dist.nrglb();
    const std::size_t len_ipiv = static_cast<std::size_t>(dist.nrloc()) + static_cast<std::size_t>(dist.nb());
    ipiv.assign(len_ipiv, 0);
    int info = 0;
    try
    {
        Block<T> a(ddla, A, dist.desc());
        const ddla::DdlaHandle_t& handle = ddla.handle();
        ddla::vector<int> pivots(handle, len_ipiv);
        ddla::pgetrf(handle, n, n, a.data(), 1, 1, dist.desc(), pivots.data(), info);
        ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle, ipiv.data(), pivots.data(), len_ipiv * sizeof(int)));
        ddla::DDLA_CHECK(ddla::ddlaSynchronize(handle));
        a.to_host();
    }
    catch (const std::exception& e)
    {
        quit_on_ddla_error("ModuleBase::ddla_linalg::getrf", e);
    }
    if (info != 0)
    {
        const std::string message = "LibDDLA pgetrf found a singular matrix, info = " + std::to_string(info);
        ModuleBase::WARNING_QUIT("ModuleBase::ddla_linalg::getrf", message);
    }
#else
    (void)ddla;
    (void)dist;
    (void)A;
    (void)ipiv;
    quit_without_ddla("ModuleBase::ddla_linalg::getrf");
#endif // __DDLA
}

template <typename T>
void getrs(const DdlaContext& ddla,
           const BlockCyclicDistribution& dist,
           const char side,
           const int nrhs,
           T* A,
           const std::vector<int>& ipiv,
           T* B,
           const int* descB)
{
#ifdef __DDLA
    const int n = dist.nrglb();
    const char no_trans = 'N';
    try
    {
        Block<T> a(ddla, A, dist.desc());
        Block<T> b(ddla, B, descB);
        const ddla::DdlaHandle_t& handle = ddla.handle();
        ddla::vector<int> pivots(handle, ipiv.size());
        ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle, pivots.data(), ipiv.data(), ipiv.size() * sizeof(int)));
        ddla::pgetrs(ddla.handle(), side, no_trans, n, nrhs, a.data(), dist.desc(), pivots.data(), b.data(), descB);
        b.to_host();
    }
    catch (const std::exception& e)
    {
        quit_on_ddla_error("ModuleBase::ddla_linalg::getrs", e);
    }
#else
    (void)ddla;
    (void)dist;
    (void)side;
    (void)nrhs;
    (void)A;
    (void)ipiv;
    (void)B;
    (void)descB;
    quit_without_ddla("ModuleBase::ddla_linalg::getrs");
#endif // __DDLA
}

template <typename T>
void gemm(const DdlaContext* ddla,
          const char transa,
          const char transb,
          const int m,
          const int n,
          const int k,
          const typename Scalar<T>::type& alpha,
          const T* A,
          const int* descA,
          const T* B,
          const int* descB,
          const typename Scalar<T>::type& beta,
          T* C,
          const int* descC)
{
    gemm_impl(ddla, transa, transb, m, n, k, alpha, A, descA, B, descB, beta, C, descC, true);
}

template <typename T>
void gemm(const DdlaContext* ddla,
          const char transa,
          const char transb,
          const int m,
          const int n,
          const int k,
          const typename Scalar<T>::type& alpha,
          const T* A,
          const int ia,
          const int ja,
          const int* descA,
          const T* B,
          const int ib,
          const int jb,
          const int* descB,
          const typename Scalar<T>::type& beta,
          T* C,
          const int ic,
          const int jc,
          const int* descC)
{
    gemm_impl(ddla, transa, transb, m, n, k, alpha, A, ia, ja, descA, B, ib, jb, descB, beta, C, ic, jc, descC, true);
}

template <typename T>
int gesv(const DdlaContext* ddla, const int n, const int nrhs, T* A, const int* descA, T* B, const int* descB)
{
    return gesv_impl(ddla, n, nrhs, A, descA, B, descB, true);
}

template <typename T>
void geadd(const DdlaContext* ddla,
           const char trans,
           const int m,
           const int n,
           const typename Scalar<T>::type& alpha,
           const T* A,
           const int* descA,
           const typename Scalar<T>::type& beta,
           T* C,
           const int* descC)
{
    geadd_impl(ddla, trans, m, n, alpha, A, descA, beta, C, descC, true);
}

template <typename T>
void same_context(const ResidentMatrix<T>& A, const ResidentMatrix<T>& B)
{
    if (A.context() != B.context())
    {
        throw std::invalid_argument("Resident matrices must share one numerical backend context");
    }
}

template <typename T>
struct ResidentLU<T>::Impl
{
    ResidentMatrix<T>& matrix;
    int status = 0;
#ifdef __DDLA
    std::unique_ptr<ddla::vector<int>> pivots;
#endif

    explicit Impl(ResidentMatrix<T>& input) : matrix(input)
    {
        if (matrix.context() == nullptr)
        {
            throw std::invalid_argument("ResidentLU requires a LibDDLA context");
        }
#ifdef __DDLA
        const int* desc = matrix.desc();
        const int n = desc[2];
        const std::size_t pivot_count = static_cast<std::size_t>(local_rows(desc)) + desc[4];
        const ddla::DdlaHandle_t& handle = matrix.context()->handle();
        pivots.reset(new ddla::vector<int>(handle, pivot_count));
        ddla::pgetrf(matrix.context()->handle(), n, n, matrix.data(), 1, 1, desc, pivots->data(), status);
#else
        quit_without_ddla("ResidentLU");
#endif
    }
};

template <typename T>
ResidentLU<T>::ResidentLU(ResidentMatrix<T>& matrix)
{
    impl_.reset(new Impl(matrix));
}
template <typename T> ResidentLU<T>::~ResidentLU() = default;
template <typename T> int ResidentLU<T>::info() const { return impl_->status; }
template <typename T>
void ResidentLU<T>::copy_pivots_to_host(std::vector<int>& pivots) const
{
#ifdef __DDLA
    const ddla::DdlaHandle_t& handle = impl_->matrix.context()->handle();
    pivots.resize(impl_->pivots->size());
    ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle, pivots.data(), impl_->pivots->data(), pivots.size() * sizeof(int)));
    ddla::DDLA_CHECK(ddla::ddlaSynchronize(handle));
#else
    (void)pivots;
    quit_without_ddla("ResidentLU::copy_pivots_to_host");
#endif
}

template <typename T>
void ResidentLU<T>::solve(char side, int nrhs, ResidentMatrix<T>& rhs) const
{
    same_context(impl_->matrix, rhs);
    if (impl_->status != 0)
    {
        throw std::runtime_error("Cannot solve with a singular resident LU factorization");
    }
#ifdef __DDLA
    // The public pgetrs API takes a mutable factor pointer, although solves
    // reuse the factorization. Keep the owned in-place storage available.
    ResidentMatrix<T>& matrix = impl_->matrix;
    const int n = matrix.desc()[2];
    const char no_trans = 'N';
    ddla::pgetrs(matrix.context()->handle(), side, no_trans, n, nrhs, matrix.data(), matrix.desc(),
                 impl_->pivots->data(), rhs.data(), rhs.desc());
#else
    quit_without_ddla("ResidentLU::solve");
#endif
}
template <typename T>
void gemm(char ta, char tb, int m, int n, int k, const typename Scalar<T>::type& alpha,
          const ResidentMatrix<T>& A, const ResidentMatrix<T>& B,
          const typename Scalar<T>::type& beta, ResidentMatrix<T>& C)
{
    same_context(A, B);
    same_context(A, C);
    gemm_impl(A.context(), ta, tb, m, n, k, alpha, A.data(), A.desc(), B.data(), B.desc(), beta, C.data(), C.desc(), false);
}
template <typename T>
void gemm(char ta, char tb, int m, int n, int k, const typename Scalar<T>::type& alpha,
          const ResidentMatrix<T>& A, int ia, int ja, const ResidentMatrix<T>& B, int ib, int jb,
          const typename Scalar<T>::type& beta, ResidentMatrix<T>& C, int ic, int jc)
{
    same_context(A, B);
    same_context(A, C);
    gemm_impl(A.context(), ta, tb, m, n, k, alpha, A.data(), ia, ja, A.desc(), B.data(), ib, jb, B.desc(), beta,
              C.data(), ic, jc, C.desc(), false);
}
template <typename T>
void geadd(char trans, int m, int n, const typename Scalar<T>::type& alpha,
           const ResidentMatrix<T>& A, const typename Scalar<T>::type& beta, ResidentMatrix<T>& C)
{
    same_context(A, C);
    geadd_impl(A.context(), trans, m, n, alpha, A.data(), A.desc(), beta, C.data(), C.desc(), false);
}
template <typename T>
int gesv(int n, int nrhs, ResidentMatrix<T>& A, ResidentMatrix<T>& B)
{
    same_context(A, B);
    return gesv_impl(A.context(), n, nrhs, A.data(), A.desc(), B.data(), B.desc(), false);
}

template void gemm<double>(const DdlaContext*, char, char, int, int, int, const double&, const double*, const int*,
                           const double*, const int*, const double&, double*, const int*);
template void gemm<std::complex<double>>(const DdlaContext*, char, char, int, int, int, const std::complex<double>&,
                                         const std::complex<double>*, const int*, const std::complex<double>*,
                                         const int*, const std::complex<double>&, std::complex<double>*, const int*);
template void gemm<double>(const DdlaContext*, char, char, int, int, int, const double&, const double*, int, int,
                           const int*, const double*, int, int, const int*, const double&, double*, int, int,
                           const int*);
template void gemm<std::complex<double>>(const DdlaContext*, char, char, int, int, int, const std::complex<double>&,
                                         const std::complex<double>*, int, int, const int*,
                                         const std::complex<double>*, int, int, const int*,
                                         const std::complex<double>&, std::complex<double>*, int, int, const int*);
template void tran<double>(const DdlaContext*, char, int, int, const double&, const double*, const int*, const double&,
                           double*, const int*);
template void tran<std::complex<double>>(const DdlaContext*, char, int, int, const std::complex<double>&,
                                         const std::complex<double>*, const int*, const std::complex<double>&,
                                         std::complex<double>*, const int*);
template int gesv<double>(const DdlaContext*, int, int, double*, const int*, double*, const int*);
template int gesv<std::complex<double>>(const DdlaContext*, int, int, std::complex<double>*, const int*,
                                        std::complex<double>*, const int*);
template int posv<double>(const DdlaContext*, char, int, int, double*, const int*, double*, const int*);
template int posv<std::complex<double>>(const DdlaContext*, char, int, int, std::complex<double>*, const int*,
                                        std::complex<double>*, const int*);
template void geadd<double>(const DdlaContext*, char, int, int, const double&, const double*, const int*,
                            const double&, double*, const int*);
template void geadd<std::complex<double>>(const DdlaContext*, char, int, int, const std::complex<double>&,
                                          const std::complex<double>*, const int*, const std::complex<double>&,
                                          std::complex<double>*, const int*);
template void getrf<double>(const DdlaContext&, const BlockCyclicDistribution&, double*, std::vector<int>&);
template void getrf<std::complex<double>>(const DdlaContext&, const BlockCyclicDistribution&, std::complex<double>*,
                                          std::vector<int>&);
template void getrs<double>(const DdlaContext&, const BlockCyclicDistribution&, char, int, double*,
                            const std::vector<int>&, double*, const int*);
template void getrs<std::complex<double>>(const DdlaContext&, const BlockCyclicDistribution&, char, int,
                                          std::complex<double>*, const std::vector<int>&, std::complex<double>*,
                                          const int*);

template class ResidentMatrix<double>;
template class ResidentMatrix<std::complex<double>>;
template class ResidentLU<double>;
template class ResidentLU<std::complex<double>>;
template void gemm<double>(char, char, int, int, int, const double&, const ResidentMatrix<double>&, const ResidentMatrix<double>&, const double&, ResidentMatrix<double>&);
template void gemm<double>(char, char, int, int, int, const double&, const ResidentMatrix<double>&, int, int, const ResidentMatrix<double>&, int, int, const double&, ResidentMatrix<double>&, int, int);
template void geadd<double>(char, int, int, const double&, const ResidentMatrix<double>&, const double&, ResidentMatrix<double>&);
template int gesv<double>(int, int, ResidentMatrix<double>&, ResidentMatrix<double>&);
template void gemm<std::complex<double>>(char, char, int, int, int, const std::complex<double>&, const ResidentMatrix<std::complex<double>>&, const ResidentMatrix<std::complex<double>>&, const std::complex<double>&, ResidentMatrix<std::complex<double>>&);
template void gemm<std::complex<double>>(char, char, int, int, int, const std::complex<double>&, const ResidentMatrix<std::complex<double>>&, int, int, const ResidentMatrix<std::complex<double>>&, int, int, const std::complex<double>&, ResidentMatrix<std::complex<double>>&, int, int);
template void geadd<std::complex<double>>(char, int, int, const std::complex<double>&, const ResidentMatrix<std::complex<double>>&, const std::complex<double>&, ResidentMatrix<std::complex<double>>&);
template int gesv<std::complex<double>>(int, int, ResidentMatrix<std::complex<double>>&, ResidentMatrix<std::complex<double>>&);
#endif // __MPI
} // namespace ddla_linalg
} // namespace ModuleBase
