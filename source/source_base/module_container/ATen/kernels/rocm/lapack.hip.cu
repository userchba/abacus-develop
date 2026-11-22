#include <algorithm>
#include <cassert>
#include <vector>
#include <ATen/kernels/lapack.h>
#include <base/third_party/lapack.h>

#include <hip/hip_runtime.h>
#include <thrust/complex.h>
#include <hipsolver/hipsolver.h>

namespace container {
namespace kernels {


static hipsolverHandle_t hipsolver_handle = nullptr;

void createGpuSolverHandle() {
    if (hipsolver_handle == nullptr) {
        hipsolverErrcheck(hipsolverCreate(&hipsolver_handle));
    }
}

void destroyGpuSolverHandle() {
    if (hipsolver_handle != nullptr) {
        hipsolverErrcheck(hipsolverDestroy(hipsolver_handle));
        hipsolver_handle = nullptr;
    }
}

template <typename T>
__global__ void set_matrix_kernel(
    const char uplo,
    T* A,
    const int dim)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    for (int ii = tid; ii < bid + 1; ii += THREADS_PER_BLOCK) {
        if (uplo == 'L') {
            A[ii * dim + bid + 1] = static_cast<T>(0);
        }
        else {
            A[(bid + 1) * dim + ii] = static_cast<T>(0);
        }
    }
}

template <typename T>
struct set_matrix<T, DEVICE_GPU> {
    using Type = typename GetTypeThrust<T>::type;
    void operator() (
        const char& uplo,
        T* A,
        const int& dim)
    {
        set_matrix_kernel<Type><<<dim - 1, THREADS_PER_BLOCK>>>(
            uplo, reinterpret_cast<Type*>(A), dim);
    }
};

template <typename T>
struct lapack_trtri<T, DEVICE_GPU> {
    void operator()(
        const char& uplo,
        const char& diag,
        const int& dim,
        T* Mat,
        const int& lda)
    {
        // TODO: trtri is not implemented in this method yet
        // Cause the trtri in cuSolver is not stable for ABACUS!
        // hipSolverConnector::trtri(hipsolver_handle, uplo, diag, dim, Mat, lda);
        // hipSolverConnector::potri(hipsolver_handle, uplo, diag, dim, Mat, lda);
        std::vector<T> H_Mat(dim * dim, static_cast<T>(0.0));
        hipMemcpy(H_Mat.data(), Mat, sizeof(T) * H_Mat.size(), hipMemcpyDeviceToHost);
        lapack_trtri<T, DEVICE_CPU>()(uplo, diag, dim, H_Mat.data(), lda);
        hipMemcpy(Mat, H_Mat.data(), sizeof(T) * H_Mat.size(), hipMemcpyHostToDevice);
    }
};

template <typename T>
struct lapack_potrf<T, DEVICE_GPU> {
    void operator()(
        const char& uplo,
        const int& dim,
        T* Mat,
        const int& lda)
    {
        // hipSolverConnector::potrf(hipsolver_handle, uplo, dim, Mat, dim);
        std::vector<T> H_Mat(dim * dim, static_cast<T>(0.0));
        hipMemcpy(H_Mat.data(), Mat, sizeof(T) * H_Mat.size(), hipMemcpyDeviceToHost);
        lapack_potrf<T, DEVICE_CPU>()(uplo, dim, H_Mat.data(), lda);
        hipMemcpy(Mat, H_Mat.data(), sizeof(T) * H_Mat.size(), hipMemcpyHostToDevice);
    }
};

template <typename T>
struct lapack_heevd<T, DEVICE_GPU> {
    using Real = typename GetTypeReal<T>::type;
    void operator()(
        const int dim,
        T* Mat,
        const int lda,
        Real* eigen_val)
    {
        // Host fallback as before, with the current (dim, Mat, lda, eigen_val) contract.
        std::vector<T> H_Mat(static_cast<size_t>(lda) * dim);
        std::vector<Real> H_eigen_val(dim);
        hipErrcheck(hipMemcpy(H_Mat.data(), Mat, sizeof(T) * H_Mat.size(), hipMemcpyDeviceToHost));
        lapack_heevd<T, DEVICE_CPU>()(dim, H_Mat.data(), lda, H_eigen_val.data());
        hipErrcheck(hipMemcpy(Mat, H_Mat.data(), sizeof(T) * H_Mat.size(), hipMemcpyHostToDevice));
        hipErrcheck(hipMemcpy(eigen_val, H_eigen_val.data(), sizeof(Real) * H_eigen_val.size(), hipMemcpyHostToDevice));
    }
};

template <typename T>
struct lapack_heevx<T, DEVICE_GPU> {
    using Real = typename GetTypeReal<T>::type;
    void operator()(
        const int n,
        const int lda,
        const T *d_Mat,
        const int neig,
        Real *d_eigen_val,
        T *d_eigen_vec)
    {
        // Host fallback: the lowest neig eigenpairs of the lower triangle, like the CUDA kernel.
        assert(n <= lda);
        std::vector<T> H_Mat(static_cast<size_t>(lda) * n);
        std::vector<T> H_vec(static_cast<size_t>(lda) * n);
        std::vector<Real> H_val(std::max(1, n));
        hipErrcheck(hipMemcpy(H_Mat.data(), d_Mat, sizeof(T) * H_Mat.size(), hipMemcpyDeviceToHost));
        lapack_heevx<T, DEVICE_CPU>()(n, lda, H_Mat.data(), neig, H_val.data(), H_vec.data());
        hipErrcheck(hipMemcpy(d_eigen_val, H_val.data(), sizeof(Real) * neig, hipMemcpyHostToDevice));
        hipErrcheck(hipMemcpy(d_eigen_vec, H_vec.data(), sizeof(T) * static_cast<size_t>(lda) * neig, hipMemcpyHostToDevice));
    }
};

template <typename T>
struct lapack_hegvd<T, DEVICE_GPU> {
    using Real = typename GetTypeReal<T>::type;
    void operator()(
        const int dim,
        const int lda,
        T* Mat_A,
        T* Mat_B,
        Real* eigen_val,
        T *eigen_vec)
    {
        const int itype = 1;
        const char jobz = 'V';
        const char uplo = 'U';
        hipErrcheck(hipMemcpy(eigen_vec, Mat_A, sizeof(T) * dim * lda, hipMemcpyDeviceToDevice));
        hipSolverConnector::hegvd(hipsolver_handle, itype, jobz, uplo, dim, Mat_A, lda, Mat_B, lda, eigen_val);
    }
};

template <typename T>
struct lapack_getrf<T, DEVICE_GPU> {
    void operator()(
        const int& m,
        const int& n,
        T* Mat,
        const int& lda,
        int* ipiv)
    {
        // hipSolverConnector has no getrf yet: factorize on the host like trtri/potrf above.
        std::vector<T> H_Mat(static_cast<size_t>(lda) * n);
        std::vector<int> H_ipiv(std::max(1, std::min(m, n)));
        hipErrcheck(hipMemcpy(H_Mat.data(), Mat, sizeof(T) * H_Mat.size(), hipMemcpyDeviceToHost));
        lapack_getrf<T, DEVICE_CPU>()(m, n, H_Mat.data(), lda, H_ipiv.data());
        hipErrcheck(hipMemcpy(Mat, H_Mat.data(), sizeof(T) * H_Mat.size(), hipMemcpyHostToDevice));
        hipErrcheck(hipMemcpy(ipiv, H_ipiv.data(), sizeof(int) * std::min(m, n), hipMemcpyHostToDevice));
    }
};

template <typename T>
struct lapack_getrs<T, DEVICE_GPU> {
    void operator()(
        const char& trans,
        const int& n,
        const int& nrhs,
        T* A,
        const int& lda,
        const int* ipiv,
        T* B,
        const int& ldb)
    {
        // Host solve with the LU factors and pivots of lapack_getrf above.
        std::vector<T> H_A(static_cast<size_t>(lda) * n);
        std::vector<int> H_ipiv(std::max(1, n));
        std::vector<T> H_B(static_cast<size_t>(ldb) * nrhs);
        hipErrcheck(hipMemcpy(H_A.data(), A, sizeof(T) * H_A.size(), hipMemcpyDeviceToHost));
        hipErrcheck(hipMemcpy(H_ipiv.data(), ipiv, sizeof(int) * n, hipMemcpyDeviceToHost));
        hipErrcheck(hipMemcpy(H_B.data(), B, sizeof(T) * H_B.size(), hipMemcpyDeviceToHost));
        lapack_getrs<T, DEVICE_CPU>()(trans, n, nrhs, H_A.data(), lda, H_ipiv.data(), H_B.data(), ldb);
        hipErrcheck(hipMemcpy(B, H_B.data(), sizeof(T) * H_B.size(), hipMemcpyHostToDevice));
    }
};

template struct set_matrix<float,  DEVICE_GPU>;
template struct set_matrix<double, DEVICE_GPU>;
template struct set_matrix<std::complex<float>,  DEVICE_GPU>;
template struct set_matrix<std::complex<double>, DEVICE_GPU>;

template struct lapack_trtri<float,  DEVICE_GPU>;
template struct lapack_trtri<double, DEVICE_GPU>;
template struct lapack_trtri<std::complex<float>,  DEVICE_GPU>;
template struct lapack_trtri<std::complex<double>, DEVICE_GPU>;

template struct lapack_potrf<float,  DEVICE_GPU>;
template struct lapack_potrf<double, DEVICE_GPU>;
template struct lapack_potrf<std::complex<float>,  DEVICE_GPU>;
template struct lapack_potrf<std::complex<double>, DEVICE_GPU>;

template struct lapack_heevd<float,  DEVICE_GPU>;
template struct lapack_heevd<double, DEVICE_GPU>;
template struct lapack_heevd<std::complex<float>,  DEVICE_GPU>;
template struct lapack_heevd<std::complex<double>, DEVICE_GPU>;

template struct lapack_heevx<float,  DEVICE_GPU>;
template struct lapack_heevx<double, DEVICE_GPU>;
template struct lapack_heevx<std::complex<float>,  DEVICE_GPU>;
template struct lapack_heevx<std::complex<double>, DEVICE_GPU>;

template struct lapack_hegvd<float,  DEVICE_GPU>;
template struct lapack_hegvd<double, DEVICE_GPU>;
template struct lapack_hegvd<std::complex<float>,  DEVICE_GPU>;
template struct lapack_hegvd<std::complex<double>, DEVICE_GPU>;

template struct lapack_getrf<float,  DEVICE_GPU>;
template struct lapack_getrf<double, DEVICE_GPU>;
template struct lapack_getrf<std::complex<float>,  DEVICE_GPU>;
template struct lapack_getrf<std::complex<double>, DEVICE_GPU>;

template struct lapack_getrs<float,  DEVICE_GPU>;
template struct lapack_getrs<double, DEVICE_GPU>;
template struct lapack_getrs<std::complex<float>,  DEVICE_GPU>;
template struct lapack_getrs<std::complex<double>, DEVICE_GPU>;

} // namespace kernels
} // namespace container
