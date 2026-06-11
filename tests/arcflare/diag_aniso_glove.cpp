// diag_aniso_glove: Phase 1 ordering-quality probe for anisotropic PQ on ANGULAR data.
// Reads GloVe hdf5 directly, L2-normalizes (cosine == L2 / IP on unit vectors), optionally
// applies a random orthogonal rotation (SRHT-like), PQ-encodes plain vs anisotropic, and
// measures brute-PQ-ADC recall@10 vs the hdf5 cosine ground-truth neighbors.
// RECALL/ORDERING ONLY — the decisive gate on whether anisotropic helps angular.
//
// Usage: diag_aniso_glove <glove.hdf5> [nq=100] [n_db=200000] [rotate=0|1]
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

// K-parameterized ScaNN anisotropic encoder. For cosine/MIPS the "parallel" direction is
// the datapoint direction; inv_norm = 1/||x|| (==1 after normalization, but computed from
// the (possibly rotated) vector to stay general). cb layout [M][K][D_sub].
static void encode_aniso(const float* xr, const float* cb, int M, int K, int D_sub,
                         float eta, uint16_t* codes_out) {
    double nrm2 = 0.0; for (int d = 0; d < M*D_sub; ++d) nrm2 += (double)xr[d]*xr[d];
    const double inv_norm = (nrm2>1e-30)?1.0/std::sqrt(nrm2):0.0;
    std::vector<float> rn((size_t)M*K), par((size_t)M*K);
    for (int s=0;s<M;++s){ const float* sv=xr+(size_t)s*D_sub; const float* cbs=cb+(size_t)s*K*D_sub;
        for (int k=0;k<K;++k){ const float* c=cbs+(size_t)k*D_sub; double a=0,p=0;
            for (int d=0;d<D_sub;++d){ double rc=(double)sv[d]-c[d]; a+=rc*rc; p+=rc*(double)sv[d]*inv_norm; }
            rn[(size_t)s*K+k]=(float)a; par[(size_t)s*K+k]=(float)p; } }
    std::vector<uint16_t> code(M); double total_par=0;
    for (int s=0;s<M;++s){ const float* r=&rn[(size_t)s*K]; int b=0; float bv=r[0];
        for (int k=1;k<K;++k) if(r[k]<bv){bv=r[k];b=k;} code[s]=(uint16_t)b; total_par+=par[(size_t)s*K+b]; }
    if (eta>1.0f){ std::vector<int> ord(M); for(int s=0;s<M;++s)ord[s]=s;
        std::sort(ord.begin(),ord.end(),[&](int a,int b){return rn[(size_t)a*K+code[a]]>rn[(size_t)b*K+code[b]];});
        const double pcm=eta;
        for(int round=0,ch=1; ch&&round<10; ++round){ ch=0;
            for(int oi=0;oi<M;++oi){ int s=ord[oi]; const float* r=&rn[(size_t)s*K]; const float* pr=&par[(size_t)s*K];
                uint16_t cur=code[s]; double orn=r[cur],opar=pr[cur]; double bd=0; int bk=cur; double btp=total_par;
                for(int k=0;k<K;++k){ if(k==(int)cur)continue; double ntp=total_par-opar+pr[k];
                    double pnd=ntp*ntp-total_par*total_par; if(pnd>0)continue;
                    double rnd=(double)r[k]-orn; double ppd=rnd-pnd; double cd=pcm*pnd+ppd;
                    if(cd<bd){bd=cd;bk=k;btp=ntp;} }
                if(bk!=(int)cur){code[s]=(uint16_t)bk;total_par=btp;ch=1;} } } }
    for(int s=0;s<M;++s) codes_out[s]=code[s];
}

static void normalize(float* v, int D){ float n=0; for(int d=0;d<D;++d)n+=v[d]*v[d];
    if(n>1e-20f){ float inv=1.f/std::sqrtf(n); for(int d=0;d<D;++d)v[d]*=inv; } }

int main(int argc, char** argv){
    if(argc<2){ fprintf(stderr,"usage: %s <glove.hdf5> [nq] [n_db] [rotate]\n",argv[0]); return 1; }
    int NQ=(argc>2)?std::atoi(argv[2]):100;
    int NDB=(argc>3)?std::atoi(argv[3]):200000;
    int ROT=(argc>4)?std::atoi(argv[4]):0;
    auto train=h5_read_float(argv[1],"train"); auto qs=h5_read_float(argv[1],"test"); auto gt=h5_read_int(argv[1],"neighbors");
    int N=std::min(NDB,train.n_rows), D=train.n_cols, nq=std::min(NQ,qs.n_rows);
    fprintf(stderr,"GloVe: N(db)=%d D=%d nq=%d rotate=%d\n",N,D,nq,ROT);

    // Random orthogonal rotation (Gram-Schmidt on a Gaussian matrix), applied if ROT.
    std::vector<float> R; // [D*D]
    if(ROT){ std::mt19937 rng(7); std::normal_distribution<float> nd(0,1);
        R.assign((size_t)D*D,0.f); for(auto&x:R)x=nd(rng);
        for(int i=0;i<D;++i){ for(int j=0;j<i;++j){ double dp=0; for(int d=0;d<D;++d)dp+=(double)R[(size_t)i*D+d]*R[(size_t)j*D+d];
            for(int d=0;d<D;++d)R[(size_t)i*D+d]-=(float)(dp*R[(size_t)j*D+d]); }
            double nn=0; for(int d=0;d<D;++d)nn+=(double)R[(size_t)i*D+d]*R[(size_t)i*D+d]; nn=1.0/std::sqrt(nn);
            for(int d=0;d<D;++d)R[(size_t)i*D+d]*=(float)nn; } }
    auto apply=[&](const float* in, float* out){ if(!ROT){ memcpy(out,in,D*sizeof(float)); return; }
        for(int i=0;i<D;++i){ double s=0; for(int d=0;d<D;++d)s+=(double)R[(size_t)i*D+d]*in[d]; out[i]=(float)s; } };

    // DB: normalize then (optional) rotate.
    std::vector<float> db((size_t)N*D);
    for(int i=0;i<N;++i){ std::vector<float> v(train.data.data()+(size_t)i*D, train.data.data()+(size_t)(i+1)*D);
        normalize(v.data(),D); apply(v.data(), db.data()+(size_t)i*D); }
    std::vector<std::vector<float>> q(nq, std::vector<float>(D));
    for(int i=0;i<nq;++i){ std::vector<float> v(qs.row(i), qs.row(i)+D); normalize(v.data(),D); apply(v.data(), q[i].data()); }

    struct Cfg{ int K,Dsub; float eta; const char* name; };
    std::vector<Cfg> cfgs = {
        {16,2,1.0f,"4bit-Dsub2 plain"}, {16,2,2.0f,"4bit-Dsub2 aniso2"}, {16,2,4.0f,"4bit-Dsub2 aniso4"},
        {256,2,1.0f,"8bit-Dsub2 plain"}, {256,2,2.0f,"8bit-Dsub2 aniso2"}, {256,2,4.0f,"8bit-Dsub2 aniso4"}, {256,2,8.0f,"8bit-Dsub2 aniso8"},
    };
    const int k=10;
    for(auto& cf:cfgs){
        const int Ds=cf.Dsub, M=D/Ds, K=cf.K;  // D=100 -> Dsub=2 gives M=50
        std::vector<float> cb((size_t)M*K*Ds);
        for(int s=0;s<M;++s){ std::vector<float> sub((size_t)N*Ds);
            for(int i=0;i<N;++i)memcpy(sub.data()+(size_t)i*Ds, db.data()+(size_t)i*D+s*Ds, Ds*sizeof(float));
            KMeansCentering km(K,Ds,0xA15011ULL+s); km.train(sub.data(),N,262144,50);
            for(int c=0;c<K;++c)memcpy(cb.data()+((size_t)s*K+c)*Ds, km.centroid(c), Ds*sizeof(float)); }
        std::vector<uint16_t> codes((size_t)N*M); std::vector<float> dbn(N,0.f);
        #pragma omp parallel for schedule(static)
        for(int i=0;i<N;++i){ uint16_t* cd=codes.data()+(size_t)i*M;
            encode_aniso(db.data()+(size_t)i*D, cb.data(), M, K, Ds, cf.eta, cd);
            float nn=0; for(int s=0;s<M;++s){ const float* cv=cb.data()+((size_t)s*K+cd[s])*Ds; for(int d=0;d<Ds;++d)nn+=cv[d]*cv[d]; } dbn[i]=nn; }
        double tot=0; int cnt=0;
        for(int qi=0;qi<nq;++qi){ const float* qq=q[qi].data();
            std::vector<float> ip((size_t)M*K); float qn=0; for(int d=0;d<D;++d)qn+=qq[d]*qq[d];
            for(int s=0;s<M;++s){ const float* qs2=qq+s*Ds; const float* cbs=cb.data()+(size_t)s*K*Ds;
                for(int c=0;c<K;++c){ const float* cv=cbs+(size_t)c*Ds; float dp=0; for(int d=0;d<Ds;++d)dp+=qs2[d]*cv[d]; ip[(size_t)s*K+c]=dp; } }
            std::vector<std::pair<float,int>> sc(N);
            for(int i=0;i<N;++i){ const uint16_t* cd=codes.data()+(size_t)i*M; float dot=0; for(int s=0;s<M;++s)dot+=ip[(size_t)s*K+cd[s]];
                sc[i]={qn+dbn[i]-2*dot,i}; }  // squared-L2 on unit vectors == 2-2cos == cosine ranking
            std::partial_sort(sc.begin(),sc.begin()+k,sc.end(),[](auto&a,auto&b){return a.first<b.first;});
            std::unordered_set<int> g; int gc=0; for(int j=0;j<k;++j){int id=gt.row(qi)[j]; if(id<N){g.insert(id);++gc;}}
            if(!gc)continue; int hit=0; for(int j=0;j<k;++j)if(g.count(sc[j].second))++hit; tot+=(double)hit/gc; ++cnt;
        }
        printf("%-20s K=%d Dsub=%d eta=%.1f rot=%d  brute-PQ recall@%d = %.4f\n",
               cf.name,K,Ds,cf.eta,ROT,k, cnt?tot/cnt:0.0); fflush(stdout);
    }
    return 0;
}
