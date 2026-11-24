// sk_portfolio_500.c  | gcc -O3 -march=native -ffast-math -fopenmp sk_portfolio_500.c -o sk -lm
// usage: sk [gamma0=1] [T0=0.1] [nrun=1000] [seed=time]   (J,h are normalised internally by their typical field)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#define N 457
#define TAU 125
typedef unsigned long long u64;
static double J0[N*N], J[N*N], J2[N*N], h0[N], h[N];

static inline u64 rnd(u64 *s){ u64 x=*s; x^=x>>12; x^=x<<25; x^=x>>27; *s=x; return x*0x2545F4914F6CDD1DULL; }
static inline double ur(u64 *s){ return (rnd(s)>>11)*(1.0/9007199254740992.0); }
static int cmp(const void*a,const void*b){ double x=*(double*)a,y=*(double*)b; return (x>y)-(x<y); }

static void rd(const char*f,double*a,int n){
    FILE*fp=fopen(f,"r"); if(!fp){printf("Cannot open %s\n",f);exit(1);}
    for(int i=0;i<n;i++) if(fscanf(fp,"%lf",&a[i])!=1){printf("Error reading %s at %d\n",f,i);exit(1);}
    fclose(fp);
}

int main(int argc,char**argv){
    double g0=argc>1?atof(argv[1]):1.0, T0=argc>2?atof(argv[2]):0.1;
    int nrun=argc>3?atoi(argv[3]):1000;
    u64 seed=argc>4?(u64)atoll(argv[4]):(u64)time(NULL);
    rd("J_input.txt",J0,N*N); rd("h_input.txt",h0,N);
    for(int i=0;i<N;i++) J0[i*N+i]=0;
    // typical field m0 = median_i sqrt(h_i^2 + sum_j J_ij^2) -> dynamics use J/m0, h/m0
    double v[N]; for(int i=0;i<N;i++){double s=h0[i]*h0[i]; for(int j=0;j<N;j++) s+=J0[i*N+j]*J0[i*N+j]; v[i]=sqrt(s);}
    qsort(v,N,sizeof(double),cmp); double m0=v[N/2];
    for(int i=0;i<N*N;i++){ J[i]=J0[i]/m0; J2[i]=J[i]*J[i]; }
    for(int i=0;i<N;i++) h[i]=h0[i]/m0;

    signed char *S=malloc((size_t)nrun*N); double *E=malloc(nrun*sizeof(double));
#ifdef _OPENMP
    double t0=omp_get_wtime();
#else
    double t0=clock()/(double)CLOCKS_PER_SEC;
#endif
    #pragma omp parallel for schedule(dynamic)
    for(int r=0;r<nrun;r++){
        u64 st=(seed+1)*0x9E3779B97F4A7C15ULL ^ ((u64)(r+1)*0xBF58476D1CE4E5B9ULL); for(int k=0;k<8;k++) rnd(&st);
        double mz[N],Fm[N],R[N],F[N]; int perm[N]; signed char *s=S+(size_t)r*N;
        for(int i=0;i<N;i++){ mz[i]=(rnd(&st)&1)?1.0:-1.0; R[i]=0; perm[i]=i; }
        for(int i=0;i<N;i++){ double a=0; for(int j=0;j<N;j++) a+=J[i*N+j]*mz[j]; Fm[i]=a; }
        // mean-field annealing with TAP (Onsager) term, every spin updated once per sweep (random permutation)
        for(int t=0;t<=TAU;t++){
            double f=1.0-(double)t/TAU, g=g0*f, T=T0*f;
            for(int a=N-1;a>0;a--){ int b=rnd(&st)%(a+1), x=perm[a]; perm[a]=perm[b]; perm[b]=x; }
            for(int q=0;q<N;q++){
                int i=perm[q]; double fl=Fm[i]+h[i]-mz[i]*R[i], he=sqrt(fl*fl+g*g);
                double nz=he>0?(T>1e-12?tanh(he/T):1.0)*fl/he:0.0, d=nz-mz[i], dq=mz[i]*mz[i]-nz*nz;
                const double *Ji=J+i*N,*J2i=J2+i*N;
                for(int j=0;j<N;j++){ Fm[j]+=Ji[j]*d; R[j]+=J2i[j]*dq; }
                mz[i]=nz;
            }
        }
        for(int i=0;i<N;i++) s[i]=mz[i]>0?1:-1;
        // local-search polish (true J0,h0): best of single flips and opposite-sign swaps, dE=2 s_i F_i (+2 s_k F_k - 4 J_ik s_i s_k)
        for(int i=0;i<N;i++){ double a=h0[i]; for(int j=0;j<N;j++) a+=J0[i*N+j]*s[j]; F[i]=a; }
        for(;;){
            double best=-1e-9; int bi=-1,bk=-1;
            for(int i=0;i<N;i++){
                double di=2*s[i]*F[i]; if(di<best){best=di;bi=i;bk=-1;}
                for(int k=i+1;k<N;k++) if(s[i]!=s[k]){
                    double d2=di+2*s[k]*F[k]-4*J0[i*N+k]*s[i]*s[k]; if(d2<best){best=d2;bi=i;bk=k;}
                }
            }
            if(bi<0) break;
            int idx[2]={bi,bk};
            for(int m=0;m<2;m++){ int i=idx[m]; if(i<0) continue; double c=-2.0*s[i]; s[i]=-s[i];
                for(int j=0;j<N;j++) F[j]+=J0[i*N+j]*c; }
        }
        double e=0; for(int i=0;i<N;i++) e-=0.5*s[i]*(F[i]+h0[i]); E[r]=e;   // E = -sum_{i<j}J s s - sum h s
    }
#ifdef _OPENMP
    double dt=omp_get_wtime()-t0;
#else
    double dt=clock()/(double)CLOCKS_PER_SEC-t0;
#endif
    FILE*fo=fopen("spins_output_500.txt","w"); if(!fo){printf("Cannot open output\n");return 1;}
    for(int r=0;r<nrun;r++){ fprintf(fo,"%.10f",E[r]); for(int i=0;i<N;i++) fprintf(fo," %d",S[(size_t)r*N+i]); fprintf(fo,"\n"); }
    fprintf(fo,"#Execution took %f seconds.\n#Config. av. %d | gamma0 %g | T0 %g | seed %llu | final spins with |mz| != 1 (not updated in the last step, rounded to the sign): 0 of %d\n",dt,nrun,g0,T0,seed,nrun*N);
    fclose(fo); printf("%d runs in %.2f s\n",nrun,dt); return 0;
}