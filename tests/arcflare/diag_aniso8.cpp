// diag_aniso8: Phase 1 ordering-quality probe for 8-bit (K=256) PQ + ScaNN anisotropic
// at D_sub=2, vs 4-bit (K=16) and plain 8-bit. MEASURES RECALL/ORDERING ONLY (no kernel,
// no graph) — brute-force PQ-ADC top-k recall = the quantized-only ceiling.
//
// Usage: diag_aniso8 <idx_dir> <hdf5> [nq=200] [n_db=200000]
#include "NGT/ArcFlare/ArcFlareIndex.h"
#include "NGT/ArcFlare/KMeansCentering.h"
#include "hdf5_io.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_set>

using namespace NGT::ArcFlare;

// K-parameterized anisotropic encoder (generalizes gpq4_encode_anisotropic to any K).
// cb layout: [M][K][D_sub]. Returns codes[M] (uint16 to hold up to 255).
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

int main(int argc, char** argv){
    if (argc<3){ fprintf(stderr,"usage: %s <idx> <hdf5> [nq] [n_db]\n",argv[0]); return 1; }
    const std::string dir=argv[1];
    int NQ=(argc>3)?std::atoi(argv[3]):200;
    int NDB=(argc>4)?std::atoi(argv[4]):200000;
    ::ArcFlare::ArcFlareIndex idx=::ArcFlare::ArcFlareIndex::load(dir+"/aqindex"); idx.loadV2(dir);
    const int D=idx.dEff();
    const uint16_t* rf=idx.rawFlat();
    if(!rf){ fprintf(stderr,"no raw_flat\n"); return 1; }
    auto qs=h5_read_float(argv[2],"test"); auto gt=h5_read_int(argv[2],"neighbors");
    int nq=std::min(NQ,qs.n_rows), dq=qs.n_cols;
    // total DB size from raw_flat
    // (rawFlat is [N*D] fp16; we use the first NDB rows for the brute probe — recall is vs
    //  the GT neighbor ids that fall within NDB; we restrict GT to ids<NDB for fairness.)
    fprintf(stderr,"D=%d nq=%d n_db=%d\n",D,nq,NDB);

    // Rotated DB (first NDB) and queries, as float.
    std::vector<float> dbrot((size_t)NDB*D);
    for(int i=0;i<NDB;++i){ std::vector<float> v(D); for(int d=0;d<D;++d) v[d]=fp16_to_float(rf[(size_t)i*D+d]);
        idx.rotateForDiag(v.data(), D, dbrot.data()+(size_t)i*D); }
    std::vector<std::vector<float>> qrot(nq, std::vector<float>(D));
    for(int i=0;i<nq;++i) idx.rotateForDiag(qs.row(i), dq, qrot[i].data());

    struct Cfg{ int K,Dsub; float eta; const char* name; };
    std::vector<Cfg> cfgs = {
        {16,2,1.0f,"4bit-Dsub2 plain"},     // current ArcFlare gpq4 M=128
        {256,2,1.0f,"8bit-Dsub2 plain"},    // qsg2-granularity 8-bit
        {256,2,4.0f,"8bit-Dsub2 aniso4"},
        {256,4,1.0f,"8bit-Dsub4 plain"},    // lossy 8-bit (room for anisotropic)
        {256,4,4.0f,"8bit-Dsub4 aniso4"},
        {256,4,6.0f,"8bit-Dsub4 aniso6"},
        {16,4,1.0f,"4bit-Dsub4 plain"},
        {16,4,4.0f,"4bit-Dsub4 aniso4"},
    };
    const int k=10;
    for (auto& cf : cfgs){
        const int M=D/cf.Dsub, K=cf.K, Ds=cf.Dsub;
        // Train codebook on rotated DB (plain k-means per subspace).
        std::vector<float> cb((size_t)M*K*Ds);
        for(int s=0;s<M;++s){ std::vector<float> sub((size_t)NDB*Ds);
            for(int i=0;i<NDB;++i) memcpy(sub.data()+(size_t)i*Ds, dbrot.data()+(size_t)i*D+s*Ds, Ds*sizeof(float));
            KMeansCentering km(K,Ds, 0xA15011ULL+s); km.train(sub.data(),NDB,262144,50);
            for(int c=0;c<K;++c) memcpy(cb.data()+((size_t)s*K+c)*Ds, km.centroid(c), Ds*sizeof(float)); }
        // Encode DB.
        std::vector<uint16_t> codes((size_t)NDB*M);
        std::vector<float> dbnorm(NDB,0.f);
        #pragma omp parallel for schedule(static)
        for(int i=0;i<NDB;++i){ uint16_t* cd=codes.data()+(size_t)i*M;
            encode_aniso(dbrot.data()+(size_t)i*D, cb.data(), M, K, Ds, cf.eta, cd);
            float nn=0; for(int s=0;s<M;++s){ const float* cv=cb.data()+((size_t)s*K+cd[s])*Ds;
                for(int d=0;d<Ds;++d) nn+=cv[d]*cv[d]; } dbnorm[i]=nn; }
        // Brute PQ-ADC recall: for each query build per-subspace IP table, score all DB by
        // ||q||^2 + ||x||^2 - 2<q,x_pq>, top-k, recall vs GT (ids<NDB).
        double tot_recall=0; int counted=0;
        for(int qi=0; qi<nq; ++qi){
            const float* q=qrot[qi].data();
            std::vector<float> ip((size_t)M*K);  // ip[s*K+c] = <q_s, c_{s,c}>
            float qn=0; for(int d=0;d<D;++d) qn+=q[d]*q[d];
            for(int s=0;s<M;++s){ const float* qs2=q+s*Ds; const float* cbs=cb.data()+(size_t)s*K*Ds;
                for(int c=0;c<K;++c){ const float* cv=cbs+(size_t)c*Ds; float dp=0;
                    for(int d=0;d<Ds;++d) dp+=qs2[d]*cv[d]; ip[(size_t)s*K+c]=dp; } }
            std::vector<std::pair<float,int>> sc(NDB);
            for(int i=0;i<NDB;++i){ const uint16_t* cd=codes.data()+(size_t)i*M; float dot=0;
                for(int s=0;s<M;++s) dot+=ip[(size_t)s*K+cd[s]]; sc[i]={qn+dbnorm[i]-2*dot, i}; }
            std::partial_sort(sc.begin(), sc.begin()+k, sc.end(), [](auto&a,auto&b){return a.first<b.first;});
            std::unordered_set<int> gset; int gtcount=0;
            for(int j=0;j<k;++j){ int id=gt.row(qi)[j]; if(id<NDB){ gset.insert(id); ++gtcount; } }
            if(gtcount==0) continue;
            int hit=0; for(int j=0;j<k;++j) if(gset.count(sc[j].second)) ++hit;
            tot_recall += (double)hit/gtcount; ++counted;
        }
        printf("%-22s K=%d Dsub=%d eta=%.1f  brute-PQ recall@%d = %.4f\n",
               cf.name, K, Ds, cf.eta, k, counted?tot_recall/counted:0.0);
        fflush(stdout);
    }
    return 0;
}
