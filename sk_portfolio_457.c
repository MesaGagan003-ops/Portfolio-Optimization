// Mean-field (SK/TAP) quantum-annealing-like solver for the portfolio Ising problem
//   E = - sum_{i<j} J_ij s_i s_j - sum_i h_i s_i
// Compile : gcc -O3 -march=native -fopenmp sk_portfolio_500.c -o sk_portfolio -lm
// Usage   : sk_portfolio [gamma_rel] [nrun] [seed] [topk] [full]
//   gamma_rel : initial transverse field in units of the typical local field (default 1.0)
//   nrun      : independent runs (default 1000)
//   seed      : random seed (default: clock)
//   topk      : number of lowest-energy UNIQUE states written to the file (default 50)
//   full      : 0 = compact lines "E idx idx ..." (0-based indices of the +1 spins = chosen stocks) [default]
//               1 = old format "E s_1 ... s_N" (what the notebook cell 44 reads)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define N 457                      // must equal N printed by the notebook
#define TAU 125                    // annealing steps
#define JFILE "J_input.txt"
#define HFILE "h_input.txt"
#define OUTFILE "spins_output_500.txt"

static double J[N * N], J2[N * N], h0[N];

// ---------------- random numbers (xoshiro256**, one state per run: thread safe & reproducible) -------------
static uint64_t splitmix(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static inline uint64_t next(uint64_t *s) {
    uint64_t r = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
    return r;
}
static inline double unif(uint64_t *s) { return (next(s) >> 11) * (1.0 / 9007199254740992.0); }
static void seed_state(uint64_t *s, uint64_t seed) { for (int i = 0; i < 4; i++) s[i] = splitmix(&seed); }

// ---------------- input -------------------------------------------------------------------------------------
static void read_inputs(void) {
    FILE *fp = fopen(JFILE, "r");
    if (!fp) { printf("Cannot open %s\n", JFILE); exit(1); }
    for (int i = 0; i < N * N; i++)
        if (fscanf(fp, "%lf", &J[i]) != 1) { printf("Error reading %s (expected %d x %d values)\n", JFILE, N, N); exit(1); }
    fclose(fp);
    fp = fopen(HFILE, "r");
    if (!fp) { printf("Cannot open %s\n", HFILE); exit(1); }
    for (int i = 0; i < N; i++)
        if (fscanf(fp, "%lf", &h0[i]) != 1) { printf("Error reading %s (expected %d values)\n", HFILE, N); exit(1); }
    fclose(fp);
    int asym = 0;
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++)
            if (fabs(J[i * N + j] - J[j * N + i]) > 1e-6) asym = 1;
    if (asym) printf("#Warning: J is not symmetric, it is symmetrised (J+J^T)/2\n");
    for (int i = 0; i < N; i++) {
        J[i * N + i] = 0.0;
        for (int j = i + 1; j < N; j++) { double v = 0.5 * (J[i * N + j] + J[j * N + i]); J[i * N + j] = J[j * N + i] = v; }
    }
    for (int i = 0; i < N * N; i++) J2[i] = J[i] * J[i];
}

static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

// ---------------- one mean-field annealing run (every spin updated once per step, random order) --------------
static void anneal(uint64_t *rs, double gamma0, signed char *s_out) {
    double mz[N], Fm[N], V[N];     // Fm_i = sum_j J_ij mz_j ,  V_i = sum_j J_ij^2 (1 - mz_j^2)   (kept incrementally)
    int perm[N];
    for (int i = 0; i < N; i++) { mz[i] = (unif(rs) < 0.5) ? 1.0 : -1.0; perm[i] = i; }
    for (int i = 0; i < N; i++) {
        double a = 0, b = 0;
        for (int j = 0; j < N; j++) { a += J[i * N + j] * mz[j]; b += J2[i * N + j] * (1.0 - mz[j] * mz[j]); }
        Fm[i] = a; V[i] = b;
    }
    for (int t = 0; t <= TAU; t++) {
        double gamma = gamma0 * (1.0 - (double)t / TAU);
        for (int a = N - 1; a > 0; a--) { int b = (int)(next(rs) % (uint64_t)(a + 1)); int tmp = perm[a]; perm[a] = perm[b]; perm[b] = tmp; }
        for (int ii = 0; ii < N; ii++) {
            int i = perm[ii];
            double f = Fm[i] + h0[i] - mz[i] * V[i];          // TAP/Onsager reaction term
            double heff = sqrt(f * f + gamma * gamma);
            if (heff < 1e-300) continue;
            double nz = f / heff;                               // T = 0: tanh(heff/T) = 1
            double d = nz - mz[i];
            if (d != 0.0) {
                double dv = mz[i] * mz[i] - nz * nz;
                const double *Ji = J + (size_t)i * N, *J2i = J2 + (size_t)i * N;
                for (int j = 0; j < N; j++) { Fm[j] += Ji[j] * d; V[j] += J2i[j] * dv; }
                mz[i] = nz;
            }
        }
    }
    for (int i = 0; i < N; i++) s_out[i] = (mz[i] > 0) ? 1 : -1;
}

// ---------------- greedy local search: single flips + swaps (+1 <-> -1), keeps F_i = h_i + sum_j J_ij s_j -----
static inline void flip(signed char *s, double *F, int i) {
    double d = -2.0 * s[i];
    const double *Ji = J + (size_t)i * N;
    for (int j = 0; j < N; j++) F[j] += Ji[j] * d;
    s[i] = -s[i];
}
static void polish(signed char *s) {
    double F[N];
    for (int i = 0; i < N; i++) { double a = h0[i]; for (int j = 0; j < N; j++) a += J[i * N + j] * s[j]; F[i] = a; }
    for (int it = 0; it < 10000; it++) {
        double best = -1e-12; int bi = -1, bk = -1;
        for (int i = 0; i < N; i++) { double d = 2.0 * s[i] * F[i]; if (d < best) { best = d; bi = i; bk = -1; } }
        for (int i = 0; i < N; i++) {
            if (s[i] < 0) continue;
            double di = 2.0 * s[i] * F[i];
            for (int k = 0; k < N; k++) {
                if (s[k] > 0) continue;
                double d = di + 2.0 * s[k] * F[k] - 4.0 * J[i * N + k] * s[i] * s[k];
                if (d < best) { best = d; bi = i; bk = k; }
            }
        }
        if (bi < 0) break;
        flip(s, F, bi);
        if (bk >= 0) flip(s, F, bk);
    }
}

static double energy(const signed char *s) {
    double E = 0.0;
    for (int i = 0; i < N; i++) {
        double a = 0.0;
        for (int j = i + 1; j < N; j++) a += J[i * N + j] * s[j];
        E -= s[i] * (h0[i] + a);
    }
    return E;
}

static const double *g_E;
static int cmp_idx(const void *a, const void *b) { double x = g_E[*(const int *)a], y = g_E[*(const int *)b]; return (x > y) - (x < y); }

int main(int argc, char **argv) {
    double gamma_rel = 1.0;
    int nrun = 1000, topk = 50, full = 0;
    uint64_t seed = (uint64_t)time(NULL);
    if (argc > 1) gamma_rel = atof(argv[1]);
    if (argc > 2) nrun = atoi(argv[2]);
    if (argc > 3) seed = (uint64_t)atoll(argv[3]);
    if (argc > 4) topk = atoi(argv[4]);
    if (argc > 5) full = atoi(argv[5]);

    read_inputs();

    // typical local field of a random state -> gamma0 = gamma_rel * median|F|   (J and h need no manual scaling)
    double fs;
    {
        uint64_t rs[4]; seed_state(rs, seed ^ 0xABCDEF);
        double sp[N], a[N];
        for (int i = 0; i < N; i++) sp[i] = (unif(rs) < 0.5) ? 1.0 : -1.0;
        for (int i = 0; i < N; i++) { double f = h0[i]; for (int j = 0; j < N; j++) f += J[i * N + j] * sp[j]; a[i] = fabs(f); }
        qsort(a, N, sizeof(double), cmpd);
        fs = a[N / 2];
    }
    double gamma0 = gamma_rel * fs;

#ifdef _OPENMP
    double t0 = omp_get_wtime();
#else
    clock_t c0 = clock();
#endif
    signed char *S = (signed char *)malloc((size_t)nrun * N);
    double *E = (double *)malloc(sizeof(double) * nrun);
    if (!S || !E) { printf("Memory allocation failed\n"); return 1; }

    #pragma omp parallel for schedule(dynamic, 4)
    for (int n = 0; n < nrun; n++) {
        uint64_t rs[4]; seed_state(rs, seed + 0x9E3779B97F4A7C15ULL * (uint64_t)(n + 1));
        signed char *s = S + (size_t)n * N;
        anneal(rs, gamma0, s);
        polish(s);
        E[n] = energy(s);
    }
#ifdef _OPENMP
    double elapsed = omp_get_wtime() - t0;
#else
    double elapsed = (double)(clock() - c0) / CLOCKS_PER_SEC;
#endif

    // histogram: number of chosen stocks (+1 spins) per run
    int *hist = (int *)calloc(N + 1, sizeof(int));
    for (int n = 0; n < nrun; n++) { int c = 0; for (int i = 0; i < N; i++) c += (S[(size_t)n * N + i] > 0); hist[c]++; }

    int *idx = (int *)malloc(sizeof(int) * nrun);
    for (int n = 0; n < nrun; n++) idx[n] = n;
    g_E = E;
    qsort(idx, nrun, sizeof(int), cmp_idx);

    FILE *fo = fopen(OUTFILE, "w");
    if (!fo) { printf("Cannot open %s\n", OUTFILE); return 1; }
    fprintf(fo, "#Execution took %.3f seconds.\n", elapsed);
    fprintf(fo, "#runs %d | gamma_rel %g (gamma0 %g) | seed %llu | N %d | format %s | best E %.10f\n", nrun, gamma_rel, gamma0,
            (unsigned long long)seed, N, full ? "E s_1..s_N" : "E then 0-based indices of +1 spins", E[idx[0]]);
    fprintf(fo, "#chosen-stock counts (count:runs):");
    for (int c = 0; c <= N; c++) if (hist[c]) fprintf(fo, " %d:%d", c, hist[c]);
    fprintf(fo, "\n");

    int written = 0;
    int *wr = (int *)malloc(sizeof(int) * (topk > 0 ? topk : 1));
    for (int q = 0; q < nrun && written < topk; q++) {
        const signed char *s = S + (size_t)idx[q] * N;
        int dup = 0;
        for (int w = 0; w < written && !dup; w++) dup = (memcmp(s, S + (size_t)wr[w] * N, N) == 0);
        if (dup) continue;
        wr[written++] = idx[q];
        fprintf(fo, "%.10f", E[idx[q]]);
        if (full) for (int i = 0; i < N; i++) fprintf(fo, " %d", s[i]);
        else      for (int i = 0; i < N; i++) if (s[i] > 0) fprintf(fo, " %d", i);
        fprintf(fo, "\n");
    }
    fclose(fo);
    printf("%d runs in %.2f s | best E %.6f | %d unique states written to %s\n", nrun, elapsed, E[idx[0]], written, OUTFILE);
    return 0;
}
