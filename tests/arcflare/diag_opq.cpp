// diag_opq: Phase 1 ordering-quality probe for OPQ (learned rotation) vs SRHT (random)
// before the 4-bit gpq4 PQ. MEASURES ORDERING ONLY (no graph) — brute-PQ-ADC recall@10.
// Reads hdf5 train/test/neighbors directly, normalizes if angular.
//
// Configs: no-rotation, SRHT (random orthonormal), OPQ (learned via alternating Procrustes),
//          and OPQ-init-from-SRHT. All with 4-bit (K=16) D_sub=2 PQ.
//
// Usage: diag_opq <hdf5> [nq=100] [n_db=200000] [angular=0|1] [opq_iters=20]
#include "NGT/ArcFlare/KMeansCentering.h"
#include "hdf5_io.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_set>

using namespace NGT::ArcFlare;

static void normalize(float* v, int D){ float n=0; for(int d=0;d<D;++d)n+=v[d]*v[d];
    if(n>1e-20f){ float inv=1.f/std::sqrtf(n); for(int d=0;d<D;++d)v[d]*=inv; } }

// Random orthonormal DxD matrix (Gram-Schmidt on Gaussian) — stands in for SRHT (both are
// data-blind orthonormal rotations; SRHT is structured-fast but ordering-equivalent here).
static std::vector<float> random_rotation(int D, uint32_t seed){
    std::mt19937 rng(seed); std::normal_distribution<float> nd(0,1);
    std::vector<float> R((size_t)D*D); for(auto&x:R)x=nd(rng);
    for(int i=0;i<D;++i){ for(int j=0;j<i;++j){ double dp=0; for(int d=0;d<D;++d)dp+=(double)R[(size_t)i*D+d]*R[(size_t)j*D+d];
        for(int d=0;d<D;++d)R[(size_t)i*D+d]-=(float)(dp*R[(size_t)j*D+d]); }
        double nn=0; for(int d=0;d<D;++d)nn+=(double)R[(size_t)i*D+d]*R[(size_t)i*D+d]; nn=1.0/std::sqrt(nn);
        for(int d=0;d<D;++d)R[(size_t)i*D+d]*=(float)nn; }
    return R; // row i = i-th output basis vector; y = R x  => y[i] = <R_row_i, x>
}
static void apply_rot(const std::vector<float>& R, const float* x, float* y, int D){
    if(R.empty()){ memcpy(y,x,D*sizeof(float)); return; }
    for(int i=0;i<D;++i){ double s=0; for(int d=0;d<D;++d)s+=(double)R[(size_t)i*D+d]*x[d]; y[i]=(float)s; }
}

// One-sided Jacobi SVD of a general DxD matrix A -> A = U S Vt. Returns U (DxD), Vt (DxD).
// Robust for D<=256. We only need R = V * Ut (the orthogonal Procrustes solution for
// max tr(Rt A) where A = Σ x_hat x^T): R = V U^T.  (Build R = (U S Vt)-> V Ut.)
static void jacobi_svd(std::vector<double>& A, int D, std::vector<double>& U, std::vector<double>& Vt){
    // One-sided Jacobi on columns of A: orthogonalize columns by Givens rotations accumulated
    // into V; column norms -> singular values; U = A V / S.
    std::vector<double> V((size_t)D*D,0.0); for(int i=0;i<D;++i)V[(size_t)i*D+i]=1.0;
    for(int sweep=0; sweep<30; ++sweep){
        double off=0;
        for(int p=0;p<D;++p) for(int q=p+1;q<D;++q){
            double app=0,aqq=0,apq=0;
            for(int i=0;i<D;++i){ double aip=A[(size_t)i*D+p], aiq=A[(size_t)i*D+q];
                app+=aip*aip; aqq+=aiq*aiq; apq+=aip*aiq; }
            if(std::fabs(apq) < 1e-15*std::sqrt(app*aqq+1e-300)) continue;
            off+=std::fabs(apq);
            double tau=(aqq-app)/(2*apq);
            double t=(tau>=0?1.0:-1.0)/(std::fabs(tau)+std::sqrt(1+tau*tau));
            double c=1.0/std::sqrt(1+t*t), s=c*t;
            for(int i=0;i<D;++i){ double aip=A[(size_t)i*D+p], aiq=A[(size_t)i*D+q];
                A[(size_t)i*D+p]=c*aip-s*aiq; A[(size_t)i*D+q]=s*aip+c*aiq; }
            for(int i=0;i<D;++i){ double vip=V[(size_t)i*D+p], viq=V[(size_t)i*D+q];
                V[(size_t)i*D+p]=c*vip-s*viq; V[(size_t)i*D+q]=s*vip+c*viq; }
        }
        if(off<1e-12) break;
    }
    // singular values = column norms of A; U columns = A columns / sigma
    U.assign((size_t)D*D,0.0); std::vector<double> sig(D,0.0);
    for(int q=0;q<D;++q){ double n=0; for(int i=0;i<D;++i)n+=A[(size_t)i*D+q]*A[(size_t)i*D+q]; n=std::sqrt(n);
        sig[q]=n; if(n>1e-300) for(int i=0;i<D;++i)U[(size_t)i*D+q]=A[(size_t)i*D+q]/n; else U[(size_t)q*D+q]=1.0; }
    // Vt = V^T
    Vt.assign((size_t)D*D,0.0); for(int i=0;i<D;++i)for(int j=0;j<D;++j)Vt[(size_t)i*D+j]=V[(size_t)j*D+i];
}

// Train a 4-bit (K=16) D_sub=2 PQ on rotated data `rot` (N x D). Fills cb [M][K][Ds].
static void train_pq(const std::vector<float>& rot, int N, int D, int Ds, int K, std::vector<float>& cb){
    int M=D/Ds; cb.assign((size_t)M*K*Ds,0.f);
    for(int s=0;s<M;++s){ std::vector<float> sub((size_t)N*Ds);
        for(int i=0;i<N;++i) memcpy(sub.data()+(size_t)i*Ds, rot.data()+(size_t)i*D+s*Ds, Ds*sizeof(float));
        KMeansCentering km(K,Ds,0xA15011ULL+s); km.train(sub.data(),N,262144,40);
        for(int c=0;c<K;++c) memcpy(cb.data()+((size_t)s*K+c)*Ds, km.centroid(c), Ds*sizeof(float)); }
}
// Encode (L2-argmin) all rotated vecs; return codes + recon (the reconstructed rotated vec).
static void encode_pq(const std::vector<float>& rot, int N, int D, int Ds, int K,
                      const std::vector<float>& cb, std::vector<uint8_t>& codes, std::vector<float>* recon){
    int M=D/Ds; codes.assign((size_t)N*M,0);
    if(recon) recon->assign((size_t)N*D,0.f);
    #pragma omp parallel for schedule(static)
    for(int i=0;i<N;++i){ const float* x=rot.data()+(size_t)i*D; uint8_t* cd=codes.data()+(size_t)i*M;
        for(int s=0;s<M;++s){ const float* sv=x+s*Ds; const float* cbs=cb.data()+(size_t)s*K*Ds;
            float bd=1e30f; int bc=0; for(int c=0;c<K;++c){ const float* cv=cbs+(size_t)c*Ds; float d=0;
                for(int dd=0;dd<Ds;++dd){float df=sv[dd]-cv[dd];d+=df*df;} if(d<bd){bd=d;bc=c;} }
            cd[s]=(uint8_t)bc; if(recon){ const float* cv=cbs+(size_t)bc*Ds;
                for(int dd=0;dd<Ds;++dd)(*recon)[(size_t)i*D+s*Ds+dd]=cv[dd]; } } }
}

// Brute PQ-ADC recall@k of `codes`(cb) for queries `qrot` vs GT ids<N.
static double brute_recall(const std::vector<float>& qrot, int nq, const std::vector<uint8_t>& codes,
                           const std::vector<float>& cb, int N, int D, int Ds, int K,
                           const H5IntDataset& gt, int k){
    int M=D/Ds; std::vector<float> dbn(N,0.f);
    for(int i=0;i<N;++i){ const uint8_t* cd=codes.data()+(size_t)i*M; float nn=0;
        for(int s=0;s<M;++s){ const float* cv=cb.data()+((size_t)s*K+cd[s])*Ds; for(int dd=0;dd<Ds;++dd)nn+=cv[dd]*cv[dd]; } dbn[i]=nn; }
    double tot=0; int cnt=0;
    for(int qi=0;qi<nq;++qi){ const float* q=qrot.data()+(size_t)qi*D;
        std::vector<float> ip((size_t)M*K); float qn=0; for(int d=0;d<D;++d)qn+=q[d]*q[d];
        for(int s=0;s<M;++s){ const float* qs2=q+s*Ds; const float* cbs=cb.data()+(size_t)s*K*Ds;
            for(int c=0;c<K;++c){ const float* cv=cbs+(size_t)c*Ds; float dp=0; for(int dd=0;dd<Ds;++dd)dp+=qs2[dd]*cv[dd]; ip[(size_t)s*K+c]=dp; } }
        std::vector<std::pair<float,int>> sc(N);
        for(int i=0;i<N;++i){ const uint8_t* cd=codes.data()+(size_t)i*M; float dot=0; for(int s=0;s<M;++s)dot+=ip[(size_t)s*K+cd[s]]; sc[i]={qn+dbn[i]-2*dot,i}; }
        std::partial_sort(sc.begin(),sc.begin()+k,sc.end(),[](auto&a,auto&b){return a.first<b.first;});
        std::unordered_set<int> g; int gc=0; for(int j=0;j<k;++j){int id=gt.row(qi)[j]; if(id<N){g.insert(id);++gc;}}
        if(!gc)continue; int hit=0; for(int j=0;j<k;++j)if(g.count(sc[j].second))++hit; tot+=(double)hit/gc; ++cnt;
    }
    return cnt?tot/cnt:0.0;
}

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s <hdf5> [nq] [n_db] [angular] [opq_iters]\n",argv[0]); return 1; }
    int NQ=(argc>2)?std::atoi(argv[2]):100, NDB=(argc>3)?std::atoi(argv[3]):200000;
    int ANG=(argc>4)?std::atoi(argv[4]):0, ITERS=(argc>5)?std::atoi(argv[5]):20;
    auto train=h5_read_float(argv[1],"train"); auto qs=h5_read_float(argv[1],"test"); auto gt=h5_read_int(argv[1],"neighbors");
    int N=std::min(NDB,train.n_rows), D=train.n_cols, nq=std::min(NQ,qs.n_rows);
    const int Ds=2, K=16, M=D/Ds, k=10;
    fprintf(stderr,"OPQ probe: N=%d D=%d nq=%d angular=%d Ds=%d M=%d iters=%d\n",N,D,nq,ANG,Ds,M,ITERS);

    // base vectors (normalized if angular)
    std::vector<float> base((size_t)N*D);
    for(int i=0;i<N;++i){ std::vector<float> v(train.data.data()+(size_t)i*D, train.data.data()+(size_t)(i+1)*D);
        if(ANG)normalize(v.data(),D); memcpy(base.data()+(size_t)i*D, v.data(), D*sizeof(float)); }
    std::vector<float> qbase((size_t)nq*D);
    for(int i=0;i<nq;++i){ std::vector<float> v(qs.row(i),qs.row(i)+D); if(ANG)normalize(v.data(),D); memcpy(qbase.data()+(size_t)i*D,v.data(),D*sizeof(float)); }

    auto rotate_all=[&](const std::vector<float>& R, const std::vector<float>& src, int n, std::vector<float>& dst){
        dst.assign((size_t)n*D,0.f); for(int i=0;i<n;++i) apply_rot(R, src.data()+(size_t)i*D, dst.data()+(size_t)i*D, D); };
    auto run=[&](const char* name, const std::vector<float>& R){
        std::vector<float> dbrot,qrot; rotate_all(R,base,N,dbrot); rotate_all(R,qbase,nq,qrot);
        std::vector<float> cb; train_pq(dbrot,N,D,Ds,K,cb);
        std::vector<uint8_t> codes; encode_pq(dbrot,N,D,Ds,K,cb,codes,nullptr);
        double rec=brute_recall(qrot,nq,codes,cb,N,D,Ds,K,gt,k);
        printf("%-18s brute-PQ recall@%d = %.4f\n",name,k,rec); fflush(stdout); };

    // (1) no rotation, (2) random (==SRHT-class)
    run("no-rotation", {});
    std::vector<float> Rsrht=random_rotation(D,0xCAFE);
    run("SRHT(random)", Rsrht);

    // (3) OPQ: learn R via alternating Procrustes. init R = identity.
    // R is DxD, y = R x. We optimize R to minimize Σ ||R x_i - decode(encode(R x_i))||^2.
    // Procrustes step: given recon ŷ_i (in rotated space) and originals x_i, the rotation
    // that best maps x -> ŷ is from SVD of cross-cov C = Σ ŷ_i x_i^T  => C=USV^T, R=U V^T.
    int Nopq = std::min(N, 100000);  // OPQ trains on a subset for speed
    std::vector<float> Ropq((size_t)D*D,0.f); for(int i=0;i<D;++i)Ropq[(size_t)i*D+i]=1.f; // init identity
    for(int it=0; it<ITERS; ++it){
        std::vector<float> dbrot; rotate_all(Ropq, base, Nopq, dbrot);
        std::vector<float> cb; train_pq(dbrot,Nopq,D,Ds,K,cb);
        std::vector<uint8_t> codes; std::vector<float> recon; encode_pq(dbrot,Nopq,D,Ds,K,cb,codes,&recon);
        // C = Σ recon_i (base_i)^T  (DxD), double accum
        std::vector<double> C((size_t)D*D,0.0);
        for(int i=0;i<Nopq;++i){ const float* r=recon.data()+(size_t)i*D; const float* x=base.data()+(size_t)i*D;
            for(int a=0;a<D;++a){ double ra=r[a]; double* Crow=&C[(size_t)a*D]; for(int b=0;b<D;++b) Crow[b]+=ra*x[b]; } }
        std::vector<double> U,Vt; jacobi_svd(C,D,U,Vt); // C=U S Vt => R = U Vt
        for(int a=0;a<D;++a)for(int b=0;b<D;++b){ double s=0; for(int t=0;t<D;++t)s+=U[(size_t)a*D+t]*Vt[(size_t)t*D+b]; Ropq[(size_t)a*D+b]=(float)s; }
    }
    run("OPQ(learned)", Ropq);
    // (4) OPQ initialized from SRHT (R0=SRHT, then alternate)
    std::vector<float> Ros = Rsrht;
    for(int it=0; it<ITERS; ++it){
        std::vector<float> dbrot; rotate_all(Ros, base, Nopq, dbrot);
        std::vector<float> cb; train_pq(dbrot,Nopq,D,Ds,K,cb);
        std::vector<uint8_t> codes; std::vector<float> recon; encode_pq(dbrot,Nopq,D,Ds,K,cb,codes,&recon);
        std::vector<double> C((size_t)D*D,0.0);
        for(int i=0;i<Nopq;++i){ const float* r=recon.data()+(size_t)i*D; const float* x=base.data()+(size_t)i*D;
            for(int a=0;a<D;++a){ double ra=r[a]; double* Crow=&C[(size_t)a*D]; for(int b=0;b<D;++b) Crow[b]+=ra*x[b]; } }
        std::vector<double> U,Vt; jacobi_svd(C,D,U,Vt);
        for(int a=0;a<D;++a)for(int b=0;b<D;++b){ double s=0; for(int t=0;t<D;++t)s+=U[(size_t)a*D+t]*Vt[(size_t)t*D+b]; Ros[(size_t)a*D+b]=(float)s; }
    }
    run("OPQ(init SRHT)", Ros);
    return 0;
}
