#!POPCORN leaderboard qr_v2
#!POPCORN gpu B200

# 2-level blocked Householder QR with a warp-level inner panel.
# Outer WY updates keep the GEMM-friendly shape; inner panels avoid
# cublasSgeqrfBatched launch/library overhead.

import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor qr2(torch::Tensor Acm, int64_t NB, int64_t IB);
torch::Tensor qr_small(torch::Tensor Acm);
"""

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cublas_v2.h>
static cublasHandle_t H = nullptr;

__global__ void extract_V(const float* A, float* V, int b, int n, int j0, int jw, int m) {
    long idx=blockIdx.x*(long)blockDim.x+threadIdx.x, tot=(long)b*m*jw; if(idx>=tot)return;
    int c=idx%jw; long t=idx/jw; int r=t%m, i=t/m;
    V[(long)i*m*jw + r + (long)c*m] = (r<c)?0.0f:(r==c)?1.0f:A[(long)i*n*n+(j0+r)+(long)(j0+c)*n];
}
__global__ void build_M_I(const float* G, const float* tau, float* M, float* Id, int b,int n,int j0,int jw,int ldJ,long sJ){
    long idx=blockIdx.x*(long)blockDim.x+threadIdx.x, tot=(long)b*jw*jw; if(idx>=tot)return;
    int c=idx%jw; long t=idx/jw; int r=t%jw, i=t/jw; long o=(long)i*sJ+r+(long)c*ldJ;
    Id[o]=(r==c)?1.0f:0.0f;
    if(r<c)M[o]=G[o]; else if(r==c){float tv=tau[(long)i*n+j0+r]; M[o]=(tv!=0.0f)?1.0f/tv:1e30f;} else M[o]=0.0f;
}
__global__ void set_ptrs(float* base,long stride,long off,float** p,int b){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<b)p[i]=base+(long)i*stride+off; }
__device__ __forceinline__ float wsum(float v){ for(int o=16;o>0;o>>=1) v+=__shfl_down_sync(0xffffffffu,v,o); return __shfl_sync(0xffffffffu,v,0); }
__global__ void __launch_bounds__(32) panel_warp(float* A, float* tau, float* V, int n, int jo, int j, int mo, int m, int nb, int jbo, int b){
    int wid=(blockIdx.x*blockDim.x+threadIdx.x)>>5, lane=threadIdx.x&31; if(wid>=b) return;
    float* As=A+(long)wid*n*n+(long)j+(long)j*n; float* tm=tau+(long)wid*n+j;
    float* Vs=V+(long)wid*mo*jbo;
    int base=j-jo;
    for(int i=0;i<nb;i++){
        int vc=base+i;
        for(int r=lane;r<vc;r+=32) Vs[r+(long)vc*mo]=0.0f;
        float ps=0; for(int r=i+lane;r<m;r+=32){ float x=As[r+(long)i*n]; ps+=x*x; }
        float sumsq=wsum(ps); float alpha=As[i+(long)i*n]; float xn2=sumsq-alpha*alpha; float beta,t;
        if(xn2<=0.0f){ t=0.0f; beta=alpha; } else { float bn=sqrtf(sumsq); beta=(alpha>=0.0f)?-bn:bn; t=(beta-alpha)/beta; }
        if(t!=0.0f){
            float sc=1.0f/(alpha-beta);
            for(int r=i+1+lane;r<m;r+=32){
                float v=As[r+(long)i*n]*sc;
                As[r+(long)i*n]=v;
                Vs[base+r+(long)vc*mo]=v;
            }
            if(lane==0){ As[i+(long)i*n]=1.0f; Vs[vc+(long)vc*mo]=1.0f; } __syncwarp();
            for(int c=i+1;c<nb;c++){
                float w=0; for(int r=i+lane;r<m;r+=32) w+=As[r+(long)i*n]*As[r+(long)c*n]; w=wsum(w);
                for(int r=i+lane;r<m;r+=32) As[r+(long)c*n]-=t*As[r+(long)i*n]*w;
            }
            __syncwarp(); if(lane==0){ As[i+(long)i*n]=beta; tm[i]=t; }
        } else {
            for(int r=i+1+lane;r<m;r+=32) Vs[base+r+(long)vc*mo]=As[r+(long)i*n];
            if(lane==0){ tm[i]=0.0f; Vs[vc+(long)vc*mo]=1.0f; }
        }
        __syncwarp();
    }
}
__global__ void inner_apply_warp(float* A, const float* tau, int n, int j, int m, int nb, int c0, int nc, int b){
    int warp=threadIdx.x>>5, lane=threadIdx.x&31;
    long wid=(long)blockIdx.x*4+warp, tot=(long)b*nc; if(wid>=tot)return;
    int cc=wid%nc, bi=wid/nc;
    float* As=A+(long)bi*n*n+(long)j+(long)j*n;
    float* C=A+(long)bi*n*n+(long)j+(long)(c0+cc)*n;
    const float* tp=tau+(long)bi*n+j;
    for(int k=0;k<nb;k++){
        float part=0.0f;
        for(int r=k+lane;r<m;r+=32){
            float v=(r==k)?1.0f:As[r+(long)k*n];
            part += v*C[r];
        }
        float dot=wsum(part);
        float tk=tp[k];
        for(int r=k+lane;r<m;r+=32){
            float v=(r==k)?1.0f:As[r+(long)k*n];
            C[r] -= tk*v*dot;
        }
        __syncwarp();
    }
}
__global__ void inner_apply8_warp(float* A, const float* tau, int n, int j, int m, int c0, int nc, int b){
    int warp=threadIdx.x>>5, lane=threadIdx.x&31;
    long wid=(long)blockIdx.x*4+warp, tot=(long)b*nc; if(wid>=tot)return;
    int cc=wid%nc, bi=wid/nc;
    float* As=A+(long)bi*n*n+(long)j+(long)j*n;
    float* C=A+(long)bi*n*n+(long)j+(long)(c0+cc)*n;
    const float* tp=tau+(long)bi*n+j;
    #pragma unroll
    for(int k=0;k<8;k++){
        float part=0.0f;
        for(int r=k+lane;r<m;r+=32){
            float v=(r==k)?1.0f:As[r+(long)k*n];
            part += v*C[r];
        }
        float dot=wsum(part);
        float tk=tp[k];
        for(int r=k+lane;r<m;r+=32){
            float v=(r==k)?1.0f:As[r+(long)k*n];
            C[r] -= tk*v*dot;
        }
        __syncwarp();
    }
}
__global__ void inner_apply_vcache(float* A, const float* tau, const float* V, int n, int j, int m, int nb, int c0, int nc, int b, int mo, int jbo, int base){
    int nW=blockDim.x>>5, warp=threadIdx.x>>5, lane=threadIdx.x&31;
    int cpm=(nc+nW-1)/nW; int bi=blockIdx.x/cpm, chunk=blockIdx.x%cpm; if(bi>=b) return;
    const float* tp=tau+(long)bi*n+j;
    const float* Vg=V+(long)bi*mo*jbo+(long)base+(long)base*mo;
    extern __shared__ float sh[];
    float* Vs=sh;
    float* Cs=sh+(long)m*nb;
    for(long idx=threadIdx.x; idx<(long)m*nb; idx+=blockDim.x){
        int r=idx%m, k=idx/m;
        Vs[r+(long)k*m]=Vg[r+(long)k*mo];
    }
    __syncthreads();
    int cc=chunk*nW+warp; if(cc>=nc) return;
    float* C=A+(long)bi*n*n+(long)j+(long)(c0+cc)*n;
    float* Cw=Cs+(long)warp*m;
    for(int r=lane;r<m;r+=32) Cw[r]=C[r];
    __syncwarp();
    for(int k=0;k<nb;k++){
        float part=0.0f;
        for(int r=k+lane;r<m;r+=32) part += Vs[r+(long)k*m]*Cw[r];
        float dot=wsum(part);
        float tk=tp[k];
        for(int r=k+lane;r<m;r+=32) Cw[r] -= tk*Vs[r+(long)k*m]*dot;
    }
    for(int r=lane;r<m;r+=32) C[r]=Cw[r];
}
__global__ void __launch_bounds__(512,1) inner_apply_vcache8(float* A, const float* tau, const float* V, int n, int j, int m, int c0, int nc, int b, int mo, int jbo, int base){
    int warp=threadIdx.x>>5, lane=threadIdx.x&31;
    int cpm=(nc+15)>>4; int bi=blockIdx.x/cpm, chunk=blockIdx.x%cpm; if(bi>=b) return;
    const float* tp=tau+(long)bi*n+j;
    const float* Vg=V+(long)bi*mo*jbo+(long)base+(long)base*mo;
    extern __shared__ float sh[];
    float* Vs=sh;
    float* Cs=sh+(long)m*8;
    for(long idx=threadIdx.x; idx<(long)m*8; idx+=512){
        int r=idx%m, k=idx/m;
        Vs[r+(long)k*m]=Vg[r+(long)k*mo];
    }
    __syncthreads();
    int cc=(chunk<<4)+warp; if(cc>=nc) return;
    float* C=A+(long)bi*n*n+(long)j+(long)(c0+cc)*n;
    float* Cw=Cs+(long)warp*m;
    for(int r=lane;r<m;r+=32) Cw[r]=C[r];
    #pragma unroll
    for(int k=0;k<8;k++){
        float part=0.0f;
        for(int r=k+lane;r<m;r+=32) part += Vs[r+(long)k*m]*Cw[r];
        float dot=wsum(part);
        float tk=tp[k];
        for(int r=k+lane;r<m;r+=32) Cw[r] -= tk*Vs[r+(long)k*m]*dot;
    }
    for(int r=lane;r<m;r+=32) C[r]=Cw[r];
}
__global__ void small_ttw(const float* T,const float* W1,float* W2,int b,int jw,int nc,int ldJ,long sJ,long sW){
    long idx=blockIdx.x*(long)blockDim.x+threadIdx.x, tot=(long)b*jw*nc; if(idx>=tot)return;
    int r=idx%jw; long q=idx/jw; int c=q%nc, i=q/nc;
    const float* Ti=T+(long)i*sJ;
    const float* Wi=W1+(long)i*sW+(long)c*jw;
    float acc=0.0f;
    for(int p=0;p<=r;p++) acc += Ti[p+(long)r*ldJ]*Wi[p];
    W2[(long)i*sW+r+(long)c*jw]=acc;
}
__global__ void small_mtw_solve(const float* G,const float* tau,const float* W1,float* W2,
    int b,int n,int j0,int jw,int nc,int ldJ,long sJ,long sW){
    long idx=blockIdx.x*(long)blockDim.x+threadIdx.x, tot=(long)b*nc; if(idx>=tot)return;
    int c=idx%nc, i=idx/nc;
    const float* Gi=G+(long)i*sJ;
    const float* Wi=W1+(long)i*sW+(long)c*jw;
    float* Wo=W2+(long)i*sW+(long)c*jw;
    float y[8];
    #pragma unroll
    for(int r=0;r<8;r++){
        if(r<jw){
            float acc=Wi[r];
            #pragma unroll
            for(int p=0;p<8;p++) if(p<r) acc -= Gi[p+(long)r*ldJ]*y[p];
            y[r]=tau[(long)i*n+j0+r]*acc;
            Wo[r]=y[r];
        }
    }
}
static void wy_update(int b,int n,int j0,int jw,int m,int c0,int nc,int ldJ,long sJ,
    float* A,float* taud,float* V,float* G,float* M,float* T,float* W1,float* W2,
    float** mPp,float** iPp,int TH){
    const float one=1.0f,zero=0.0f,neg=-1.0f;
    long sV=(long)m*jw, sW=(long)jw*n, sC=(long)n*n;
    cublasSgemmStridedBatched(H,CUBLAS_OP_T,CUBLAS_OP_N,jw,jw,m,&one,V,m,sV,V,m,sV,&zero,G,ldJ,sJ,b);
    float* C = A + (long)j0 + (long)c0*n;
    cublasSgemmStridedBatched(H,CUBLAS_OP_T,CUBLAS_OP_N,jw,nc,m,&one,V,m,sV,C,n,sC,&zero,W1,jw,sW,b);
    if(jw<=8) small_mtw_solve<<<((long)b*nc+TH-1)/TH,TH>>>(G,taud,W1,W2,b,n,j0,jw,nc,ldJ,sJ,sW);
    else {
        build_M_I<<<((long)b*jw*jw+TH-1)/TH,TH>>>(G,taud,M,T,b,n,j0,jw,ldJ,sJ);
        cublasStrsmBatched(H,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,jw,jw,&one,mPp,ldJ,iPp,ldJ,b);
        cublasSgemmStridedBatched(H,CUBLAS_OP_T,CUBLAS_OP_N,jw,nc,jw,&one,T,ldJ,sJ,W1,jw,sW,&zero,W2,jw,sW,b);
    }
    cublasSgemmStridedBatched(H,CUBLAS_OP_N,CUBLAS_OP_N,m,nc,jw,&neg,V,m,sV,W2,jw,sW,&one,C,n,sC,b);
}

__global__ void __launch_bounds__(896,1) qr_small_kernel(float* A, float* tau, int n, int b){
    int bi=blockIdx.x; if(bi>=b) return;
    int tid=threadIdx.x, nt=blockDim.x, lane=tid&31, warp=tid>>5, nwarp=nt>>5;
    extern __shared__ float sh[];
    float* sA=sh;
    float* scr=sh+(long)n*n;
    float* Ag=A+(long)bi*n*n;
    float* tg=tau+(long)bi*n;
    for(long idx=tid; idx<(long)n*n; idx+=nt) sA[idx]=Ag[idx];
    __syncthreads();
    for(int j=0;j<n;j++){
        float* col=sA+(long)j*n;
        float ps=0.0f;
        for(int r=j+tid;r<n;r+=nt){ float x=col[r]; ps+=x*x; }
        for(int o=16;o>0;o>>=1) ps+=__shfl_down_sync(0xffffffffu,ps,o);
        if(lane==0) scr[warp]=ps;
        __syncthreads();
        if(warp==0){
            float s=(lane<nwarp)?scr[lane]:0.0f;
            for(int o=16;o>0;o>>=1) s+=__shfl_down_sync(0xffffffffu,s,o);
            if(lane==0) scr[0]=s;
        }
        __syncthreads();
        float sumsq=scr[0];
        __syncthreads();
        float alpha=col[j], xn2=sumsq-alpha*alpha, beta, t;
        if(xn2<=0.0f){ t=0.0f; beta=alpha; }
        else { float bn=sqrtf(sumsq); beta=(alpha>=0.0f)?-bn:bn; t=(beta-alpha)/beta; }
        if(t!=0.0f){
            float sc=1.0f/(alpha-beta);
            for(int r=j+1+tid;r<n;r+=nt) col[r]*=sc;
            __syncthreads();
            for(int c=j+1+warp;c<n;c+=nwarp){
                float* cc=sA+(long)c*n;
                float w=(lane==0)?cc[j]:0.0f;
                for(int r=j+1+lane;r<n;r+=32) w+=col[r]*cc[r];
                w=wsum(w);
                if(lane==0) cc[j]-=t*w;
                for(int r=j+1+lane;r<n;r+=32) cc[r]-=t*col[r]*w;
            }
            __syncthreads();
            if(tid==0){ col[j]=beta; tg[j]=t; }
            __syncthreads();
        } else {
            if(tid==0) tg[j]=0.0f;
            __syncthreads();
        }
    }
    for(long idx=tid; idx<(long)n*n; idx+=nt) Ag[idx]=sA[idx];
}
torch::Tensor qr_small(torch::Tensor Acm){
    const int b=Acm.size(0), n=Acm.size(1);
    auto tau=torch::empty({(long)b,(long)n}, Acm.options());
    const int nt=896;
    size_t shp=((size_t)n*n + (nt>>5))*sizeof(float);
    static int once=0;
    if(!once){
        cudaFuncSetAttribute(qr_small_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 200*1024);
        once=1;
    }
    qr_small_kernel<<<b,nt,shp>>>(Acm.data_ptr<float>(),tau.data_ptr<float>(),n,b);
    return tau;
}
torch::Tensor qr2(torch::Tensor Acm,int64_t NB_,int64_t IB_){
    TORCH_CHECK(Acm.is_cuda()&&Acm.scalar_type()==torch::kFloat32&&Acm.dim()==3,"bad");
    const int b=Acm.size(0),n=Acm.size(1),NB=(int)NB_,IB=(int)IB_;
    Acm=Acm.contiguous(); if(H==nullptr)cublasCreate(&H);
    static int once=0; if(!once){
        cudaFuncSetAttribute(inner_apply_vcache, cudaFuncAttributeMaxDynamicSharedMemorySize, 200*1024);
        cudaFuncSetAttribute(inner_apply_vcache8, cudaFuncAttributeMaxDynamicSharedMemorySize, 200*1024);
        once=1;
    }
    auto fo=Acm.options(); auto po=torch::TensorOptions().dtype(torch::kInt64).device(Acm.device());
    auto tau=torch::empty({b,n},fo);
    auto Vb=torch::empty({(long)b*n*NB},fo),Gb=torch::empty({(long)b*NB*NB},fo);
    auto Mb=torch::empty({(long)b*NB*NB},fo),Tb=torch::empty({(long)b*NB*NB},fo);
    auto W1b=torch::empty({(long)b*NB*n},fo),W2b=torch::empty({(long)b*NB*n},fo);
    auto mP=torch::empty({b},po),iP=torch::empty({b},po);
    float **mPp=(float**)mP.data_ptr<int64_t>(),**iPp=(float**)iP.data_ptr<int64_t>();
    float *A=Acm.data_ptr<float>(),*taud=tau.data_ptr<float>(),*V=Vb.data_ptr<float>();
    float *G=Gb.data_ptr<float>(),*M=Mb.data_ptr<float>(),*T=Tb.data_ptr<float>(),*W1=W1b.data_ptr<float>(),*W2=W2b.data_ptr<float>();
    const int TH=256; int info=0;
    const int GB=(b+TH-1)/TH, ldJ=NB; const long sJ=(long)NB*NB;
    set_ptrs<<<GB,TH>>>(M,sJ,0,mPp,b);
    set_ptrs<<<GB,TH>>>(T,sJ,0,iPp,b);
    for(int jo=0; jo<n; jo+=NB){
        int jbo=(n-jo<NB)?(n-jo):NB; int mo=n-jo;
        for(int ji=jo; ji<jo+jbo; ji+=IB){
            int jbi=(jo+jbo-ji<IB)?(jo+jbo-ji):IB; int mi=n-ji;
            int icols=jo+jbo-(ji+jbi);
            panel_warp<<<b,32>>>(A,taud,V,n,jo,ji,mo,mi,jbi,jbo,b);
            if(icols>0){
                int nW=16, cpm=(icols+nW-1)/nW; size_t shp=(size_t)(mi*jbi+nW*mi)*sizeof(float);
                if(jbi==8){
                    inner_apply_vcache8<<<(long)b*cpm,nW*32,shp>>>(A,taud,V,n,ji,mi,ji+jbi,icols,b,mo,jbo,ji-jo);
                } else {
                    inner_apply_vcache<<<(long)b*cpm,nW*32,shp>>>(A,taud,V,n,ji,mi,jbi,ji+jbi,icols,b,mo,jbo,ji-jo);
                }
            }
        }
        int ocols=n-(jo+jbo);
        if(ocols>0) wy_update(b,n,jo,jbo,mo, jo+jbo,ocols,ldJ,sJ, A,taud,V,G,M,T,W1,W2,mPp,iPp,TH);
    }
    return tau;
}
"""

_mod = load_inline(name="qrwpan_codex_small226_896lb_vcomp_cacheC8spec_nosync_nb64", cpp_sources=[CPP_SRC], cuda_sources=[CUDA_SRC],
                   functions=["qr2","qr_small"], extra_cflags=["-O3"], extra_cuda_cflags=["-O3","--use_fast_math"],
                   extra_ldflags=["-lcublas"], verbose=False)

_NB, _IB, _MAXN = 64, 8, 2048

def custom_kernel(data: input_t) -> output_t:
    if (data.is_cuda and data.dtype == torch.float32 and data.dim() == 3
            and data.size(1) == data.size(2) and 0 < data.size(1) <= _MAXN):
        Acm = data.transpose(-2, -1).contiguous()
        if data.size(1) <= 226:
            tau = _mod.qr_small(Acm)
        else:
            tau = _mod.qr2(Acm, _NB, _IB)
        H = Acm.transpose(-2, -1)
        return H, tau
    return torch.geqrf(data)
