#!POPCORN leaderboard qr_v2
#!POPCORN gpu B200

# FUSED blocked Householder QR: whole panel loop inside ONE C++ call (no Python per panel).
# Column-major (cuBLAS-native). geqrfBatched panel + closed-form T (trsm) + strided-batched
# GEMM trailing. Mirrors CPU-validated blocked_householder.py. FP32 first.
# (NOTE: the prepare scanner rejects that one CUDA-queue word anywhere, even in comments.)

import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor qr_blocked(torch::Tensor Acm, int64_t nb);
"""

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cublas_v2.h>
static cublasHandle_t H = nullptr;

__global__ void extract_V(const float* A, float* V, int b, int n, int j, int jb, int msub) {
    long idx = blockIdx.x*(long)blockDim.x + threadIdx.x, tot=(long)b*msub*jb;
    if (idx>=tot) return;
    int c=idx%jb; long t=idx/jb; int r=t%msub, i=t/msub;
    float v = (r<c)?0.0f : (r==c)?1.0f : A[(long)i*n*n + (j+r) + (long)(j+c)*n];
    V[(long)i*msub*jb + r + (long)c*msub] = v;
}
__global__ void build_M_I(const float* G, const float* tau, float* M, float* Id,
                          int b, int n, int j, int jb) {
    long idx = blockIdx.x*(long)blockDim.x + threadIdx.x, tot=(long)b*jb*jb;
    if (idx>=tot) return;
    int c=idx%jb; long t=idx/jb; int r=t%jb, i=t/jb;
    long off=(long)i*jb*jb + r + (long)c*jb;
    Id[off] = (r==c)?1.0f:0.0f;
    if (r<c) M[off]=G[off];
    else if (r==c){ float tv=tau[(long)i*n + j + r]; M[off]=(tv!=0.0f)?1.0f/tv:1e30f; }
    else M[off]=0.0f;
}
__global__ void set_ptrs(float* base, long stride, long off, float** p, int b) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<b) p[i]=base + (long)i*stride + off;
}

torch::Tensor qr_blocked(torch::Tensor Acm, int64_t nb_) {
    TORCH_CHECK(Acm.is_cuda() && Acm.scalar_type()==torch::kFloat32 && Acm.dim()==3, "bad A");
    const int b=Acm.size(0), n=Acm.size(1), nb=(int)nb_;
    Acm = Acm.contiguous();
    if (H==nullptr) cublasCreate(&H);
    auto fo=Acm.options(); auto po=torch::TensorOptions().dtype(torch::kInt64).device(Acm.device());
    auto tau=torch::zeros({b,n},fo);
    auto Vb=torch::empty({(long)b*n*nb},fo), Gb=torch::empty({(long)b*nb*nb},fo);
    auto Mb=torch::empty({(long)b*nb*nb},fo), Tb=torch::empty({(long)b*nb*nb},fo);
    auto W1b=torch::empty({(long)b*nb*n},fo), W2b=torch::empty({(long)b*nb*n},fo);
    auto aP=torch::empty({b},po),tP=torch::empty({b},po),mP=torch::empty({b},po),iP=torch::empty({b},po);
    float **aPp=(float**)aP.data_ptr<int64_t>(), **tPp=(float**)tP.data_ptr<int64_t>();
    float **mPp=(float**)mP.data_ptr<int64_t>(), **iPp=(float**)iP.data_ptr<int64_t>();
    float *A=Acm.data_ptr<float>(), *taud=tau.data_ptr<float>(), *V=Vb.data_ptr<float>();
    float *G=Gb.data_ptr<float>(), *M=Mb.data_ptr<float>(), *T=Tb.data_ptr<float>();
    float *W1=W1b.data_ptr<float>(), *W2=W2b.data_ptr<float>();
    const float one=1.0f, zero=0.0f, neg=-1.0f; const int TH=256; const int GB=(b+TH-1)/TH;
    set_ptrs<<<GB,TH>>>(M,(long)nb*nb,0,mPp,b);
    set_ptrs<<<GB,TH>>>(T,(long)nb*nb,0,iPp,b);
    for (int j=0; j<n; j+=nb) {
        int jb = (n-j<nb)?(n-j):nb;
        int msub = n-j; int info=0;
        set_ptrs<<<GB,TH>>>(A,(long)n*n,(long)j+(long)j*n,aPp,b);
        set_ptrs<<<GB,TH>>>(taud,(long)n,(long)j,tPp,b);
        cublasSgeqrfBatched(H, msub, jb, aPp, n, tPp, &info, b);
        if (j+jb<n) {                                   // jb==nb here
            int ncols=n-j-jb; long sV=(long)msub*jb, sJ=(long)jb*jb, sW=(long)jb*n;
            long totV=(long)b*msub*jb, totG=(long)b*jb*jb;
            extract_V<<<(totV+TH-1)/TH,TH>>>(A,V,b,n,j,jb,msub);
            cublasSgemmStridedBatched(H,CUBLAS_OP_T,CUBLAS_OP_N, jb,jb,msub,
                &one, V,msub,sV, V,msub,sV, &zero, G,jb,sJ, b);          // G=V^T V
            build_M_I<<<(totG+TH-1)/TH,TH>>>(G,taud,M,T,b,n,j,jb);       // M, Id->T
            cublasStrsmBatched(H,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT, jb,jb,&one, mPp,jb, iPp,jb, b);    // T=M^{-1}
            float* C = A + (long)j + (long)(j+jb)*n; long sC=(long)n*n;
            cublasSgemmStridedBatched(H,CUBLAS_OP_T,CUBLAS_OP_N, jb,ncols,msub,
                &one, V,msub,sV, C,n,sC, &zero, W1,jb,sW, b);            // W1=V^T C
            cublasSgemmStridedBatched(H,CUBLAS_OP_T,CUBLAS_OP_N, jb,ncols,jb,
                &one, T,jb,sJ, W1,jb,sW, &zero, W2,jb,sW, b);            // W2=T^T W1
            cublasSgemmStridedBatched(H,CUBLAS_OP_N,CUBLAS_OP_N, msub,ncols,jb,
                &neg, V,msub,sV, W2,jb,sW, &one, C,n,sC, b);            // C -= V W2
        }
    }
    return tau;
}
"""

_mod = load_inline(name="qr_fused", cpp_sources=[CPP_SRC], cuda_sources=[CUDA_SRC],
                   functions=["qr_blocked"], extra_cflags=["-O3"], extra_cuda_cflags=["-O3"],
                   extra_ldflags=["-lcublas"], verbose=False)

_NB = 16
_MAXN = 1024

def custom_kernel(data: input_t) -> output_t:
    if (data.is_cuda and data.dtype == torch.float32 and data.dim() == 3
            and data.size(1) == data.size(2) and 0 < data.size(1) <= _MAXN):
        Acm = data.transpose(-2, -1).contiguous()
        tau = _mod.qr_blocked(Acm, _NB)
        H = Acm.transpose(-2, -1).contiguous()
        return (H, tau)
    return torch.geqrf(data)
