#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define N 457                     // number of stocks = number of spins (must equal N of the notebook; the notebook prints it)
#define tau 125
#define NRUN_DEFAULT 10000
#define JFILE "J_input.txt"       // N x N coupling matrix J_C (written by the notebook)
#define HFILE "h_input.txt"       // N field values h_C   (written by the notebook)      <<< CHANGED: new input
#define OUTFILE "spins_output.txt"// one line per run: energy s_1 ... s_N                <<< CHANGED: new output
//#define M_PI 3.14159265359
float **J;
static double J0[N][N];           // CHANGED: couplings in double precision (used for the energies written to the output file)
double hf[N];                     // CHANGED: field h_i of every spin (the original code had one scalar h = 0)

// Function to generate Gaussian-distributed random numbers using Box-Muller method
double gaussian_random() {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return z0;
}

// Function to initialize the J_ij matrix with Gaussian values
// (kept as it was, but no longer called)
void initialize_J() {
    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            if (i == j) {
                J[i][j] = 0.0;  // No self-interaction
            } else {
                double Jij = gaussian_random() / sqrt(N);  // Variance 1/N
                J[i][j] = Jij;
                J[j][i] = Jij;  // Ensure symmetry: J_ij = J_ji
            }
        }
    }
}

// read the J_ij matrix from a text file (N rows x N columns, whitespace separated)
void read_J(const char *fname) {
    FILE *fp = fopen(fname, "r");
    if (fp == NULL) {
        printf("Cannot open input file %s\n", fname);
        exit(1);
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double v;
            if (fscanf(fp, "%lf", &v) != 1) {
                printf("Error reading J[%d][%d] from %s (expected %d x %d values)\n", i, j, fname, N, N);
                fclose(fp);
                exit(1);
            }
            J[i][j] = (float)v;
            J0[i][j] = v;
        }
    }
    fclose(fp);

    // Warn if the matrix is not symmetric
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++)
            if (fabs(J[i][j] - J[j][i]) > 1e-6)
                printf("#Warning: J is not symmetric at (%d,%d)\n", i, j);
}

// CHANGED (new): read the N field values h_i from a text file (whitespace separated)
void read_h(const char *fname) {
    FILE *fp = fopen(fname, "r");
    if (fp == NULL) {
        printf("Cannot open input file %s\n", fname);
        exit(1);
    }
    for (int i = 0; i < N; i++) {
        double v;
        if (fscanf(fp, "%lf", &v) != 1) {
            printf("Error reading h[%d] from %s (expected %d values)\n", i, fname, N);
            fclose(fp);
            exit(1);
        }
        hf[i] = v;
    }
    fclose(fp);
}

// usage:  sk_portfolio [gamma0] [scale] [nrun] [seed]
//   gamma0 : initial transverse field (default 1.0, the value of the original code)
//   scale  : J and h are multiplied by this number inside the dynamics (default 1.0 = unchanged)
//   nrun   : number of independent runs (default 10000; compile with -O2: for N = 457 one run takes about 0.02 s, try 200 first)
//   seed   : random seed (default: the clock)
int main(int argc, char **argv) {
    clock_t start, end;
    start = clock();

    int TMAX = tau;
    double mz[N], mx[N], gamma, gamma0 = 1.0, T, T0, changex, changez, heff, sum, q2, scale = 1.0;
    int nrun = NRUN_DEFAULT;
    unsigned int seed = (unsigned int)time(NULL);
    if (argc > 1) gamma0 = atof(argv[1]);
    if (argc > 2) scale = atof(argv[2]);
    if (argc > 3) nrun = atoi(argv[3]);
    if (argc > 4) seed = (unsigned int)atoi(argv[4]);
    srand(seed);

    // Allocate memory for J matrix
    J = (float**)malloc(N * sizeof(float*));
    if (J == NULL) {
        printf("Memory allocation failed for J matrix.\n");
        return 1;
    }
    for (int i = 0; i < N; i++) {
        J[i] = (float*)malloc(N * sizeof(float));
        if (J[i] == NULL) {
            printf("Memory allocation failed for J[%d].\n", i);
            for (int k = 0; k <= i; k++) free(J[k]);
            free(J);
            return 1;
        }
    }

    read_J(JFILE);   // load J_ij from the input file (once)
    read_h(HFILE);   // CHANGED: load h_i from the input file (once)

    // CHANGED: keep the UNSCALED h (and J0, read in double precision) to compute the energies written to the output file
    double h0v[N];
    for (int i = 0; i < N; i++) h0v[i] = hf[i];
    // optional scaling of the couplings and fields used in the dynamics
    for (int i = 0; i < N; i++) {
        hf[i] *= scale;
        for (int j = 0; j < N; j++) J[i][j] = (float)(J[i][j] * scale);
    }

    FILE *fo = fopen(OUTFILE, "w");
    if (fo == NULL) {
        printf("Cannot open output file %s\n", OUTFILE);
        return 1;
    }

    // Initialize parameters
    T0 = 0.;

    int nonbinary = 0;
    for (int n = 0; n < nrun; n++) {
        // Initialize spins
        for (int i = 0; i < N; i++) {
            mz[i] = (rand() / (double)RAND_MAX < 0.5) ? 1.0 : -1.0;
            mx[i] = (rand() / (double)RAND_MAX < 0.5) ? 0.0 :  0.0;
        }

        for (int t = 0; t < TMAX+1; t++) {
            T = T0 * (1.0 - (float)t / tau);
            gamma = gamma0 * (1.0 - (float)t / (tau));

            for (int ii = 0; ii < N; ii++) {
                int i = rand() % N;  // Random index in range [0, N-1]
                sum = 0;
                q2 = 0;

                for (int j = 0; j < N; j++) {
                    if (i != j){ sum += J[i][j] * mz[j];}
                    q2 += mz[j] * mz[j];
                }
                q2 = q2 / N;

                // CHANGED: the field of spin i is hf[i] (the original code used the scalar h = 0)
                heff = sqrt((sum + hf[i] - (1 - q2) * mz[i]) * (sum + hf[i] - (1 - q2) * mz[i]) + gamma * gamma);
                changez = tanh(heff / T) * (sum + hf[i] - (1 - q2) * mz[i]) / heff;
                changex = tanh(heff / T) * (gamma) / heff;

                if (changez * changez > 0) {
                    mz[i] = changez;
                }
                if (changex * changex > 0) {
                    mx[i] = changex;
                }
            }
        }

        // CHANGED: after the last step (gamma = 0) every mz is +-1; write the spins and the energy of the run
        //   energy  E_C = - sum_{i<j} J_ij s_i s_j - sum_i h_i s_i      (UNSCALED J and h)
        int s[N];
        for (int i = 0; i < N; i++) {
            s[i] = (mz[i] > 0) ? 1 : -1;
            if (fabs(fabs(mz[i]) - 1.0) > 1e-6) nonbinary++;
        }
        double E = 0.0;
        for (int i = 0; i < N; i++) {
            E -= h0v[i] * s[i];
            for (int j = i + 1; j < N; j++) E -= J0[i][j] * s[i] * s[j];
        }
        fprintf(fo, "%.10f", E);
        for (int i = 0; i < N; i++) fprintf(fo, " %d", s[i]);
        fprintf(fo, "\n");
    } //nrun loop

    end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC; // in seconds
    fprintf(fo, "#Execution took %f seconds.\n", time_taken);
    fprintf(fo, "#Config. av. %d | gamma0 %g | scale %g | seed %u | final spins with |mz| != 1 (not updated in the last step, rounded to the sign): %d of %d\n", nrun, gamma0, scale, seed, nonbinary, nrun * N);
    fclose(fo);
    printf("%d runs written to %s in %.2f s (gamma0 = %g, scale = %g, seed = %u)\n", nrun, OUTFILE, time_taken, gamma0, scale, seed);
    return 0;
}
