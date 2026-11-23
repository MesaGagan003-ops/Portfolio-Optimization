/*
 * Cardinality-constrained mean-field annealing for the 457-spin Ising model.
 *
 * Input convention (same as the original C program):
 *   E(s) = - sum_{i<j} J[i][j] s[i] s[j] - sum_i h[i] s[i]
 * where s_i is +/-1, J is symmetric with zero diagonal, and h is a vector.
 *
 * The portfolio cardinality is enforced in the continuous mean-field state:
 *   sum_i m_i = 2*k - N,
 * by solving for a Lagrange multiplier at every sweep. Final spins are
 * obtained by setting the k largest magnetisations to +1 and the rest to -1.
 * Thus EVERY output configuration has exactly k selected assets.
 *
 * This is a classical, finite-temperature mean-field annealing heuristic;
 * it is not a physical quantum annealer or an exact optimiser.
 *
 * Build:
 *   gcc -O3 -std=c11 -Wall -Wextra -pedantic \
 *       sk_portfolio_500_meanfield_fixed.c -lm -o sk_portfolio_500_meanfield_fixed
 *
 * Run from the directory containing J_input.txt and h_input.txt:
 *   ./sk_portfolio_500_meanfield_fixed [k] [gamma0] [T0] [Tmin] [sweeps] [nrun] [seed]
 * Example:
 *   ./sk_portfolio_500_meanfield_fixed 10 1.0 0.20 0.005 500 100 12345
 *
 * Output: spins_output_500_meanfield_fixed.txt
 * Each data row is: energy s_0 s_1 ... s_(N-1)
 */

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 457
#define JFILE "J_input.txt"
#define HFILE "h_input.txt"
#define OUTFILE "spins_output_500_meanfield_fixed.txt"
#define DEFAULT_K 10
#define DEFAULT_GAMMA0 1.0
#define DEFAULT_T0 0.20
#define DEFAULT_TMIN 0.005
#define DEFAULT_SWEEPS 500
#define DEFAULT_NRUN 100
#define LAMBDA_BISECTIONS 80
#define DAMPING 0.50

static double J[N][N];
static double h[N];
static double m[N], m_new[N], local_field[N];

static double urand01(void) {
    return (rand() + 0.5) / (RAND_MAX + 1.0);
}

static void read_inputs(void) {
    FILE *fp = fopen(JFILE, "r");
    if (!fp) {
        fprintf(stderr, "Cannot open %s: %s\n", JFILE, strerror(errno));
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (fscanf(fp, "%lf", &J[i][j]) != 1) {
                fprintf(stderr, "Invalid/missing J[%d][%d] in %s\n", i, j, JFILE);
                fclose(fp);
                exit(EXIT_FAILURE);
            }
        }
    }
    fclose(fp);

    fp = fopen(HFILE, "r");
    if (!fp) {
        fprintf(stderr, "Cannot open %s: %s\n", HFILE, strerror(errno));
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < N; ++i) {
        if (fscanf(fp, "%lf", &h[i]) != 1) {
            fprintf(stderr, "Invalid/missing h[%d] in %s\n", i, HFILE);
            fclose(fp);
            exit(EXIT_FAILURE);
        }
    }
    fclose(fp);

    for (int i = 0; i < N; ++i) {
        if (fabs(J[i][i]) > 1e-10) {
            fprintf(stderr, "J diagonal is not zero at index %d\n", i);
            exit(EXIT_FAILURE);
        }
        for (int j = i + 1; j < N; ++j) {
            if (fabs(J[i][j] - J[j][i]) > 1e-8) {
                fprintf(stderr, "J is not symmetric at (%d,%d)\n", i, j);
                exit(EXIT_FAILURE);
            }
        }
    }
}

/* Stable mean-field response for longitudinal field z and transverse field g. */
static double response(double z, double g, double temperature) {
    const double e = hypot(z, g);
    if (e < 1e-14) return 0.0;
    return (z / e) * tanh(e / temperature);
}

/* Sum of mean-field magnetisations for a trial Lagrange multiplier. */
static double magnetisation_sum(double lambda, double gamma, double temperature) {
    double total = 0.0;
    for (int i = 0; i < N; ++i) {
        const double z = local_field[i] - lambda;
        total += response(z, gamma, temperature);
    }
    return total;
}

/*
 * Find lambda so that sum_i m_i = target_M. The response is monotone
 * decreasing in lambda, so bisection is robust.
 */
static double solve_lambda(double target_M, double gamma, double temperature) {
    double lo = -1.0, hi = 1.0;
    int expand = 0;
    while (magnetisation_sum(lo, gamma, temperature) < target_M && expand++ < 200)
        lo *= 2.0;
    expand = 0;
    while (magnetisation_sum(hi, gamma, temperature) > target_M && expand++ < 200)
        hi *= 2.0;

    for (int it = 0; it < LAMBDA_BISECTIONS; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (magnetisation_sum(mid, gamma, temperature) > target_M)
            lo = mid;
        else
            hi = mid;
    }
    return 0.5 * (lo + hi);
}

/* Euclidean projection onto {-1 <= x_i <= 1, sum_i x_i = target}. */
static void project_box_sum(double values[N], double target) {
    double lo = -4.0, hi = 4.0;
    for (int it = 0; it < 100; ++it) {
        const double shift = 0.5 * (lo + hi);
        double sum = 0.0;
        for (int i = 0; i < N; ++i) {
            double v = values[i] + shift;
            if (v > 1.0) v = 1.0;
            if (v < -1.0) v = -1.0;
            sum += v;
        }
        if (sum < target) lo = shift;
        else hi = shift;
    }
    const double shift = 0.5 * (lo + hi);
    for (int i = 0; i < N; ++i) {
        values[i] += shift;
        if (values[i] > 1.0) values[i] = 1.0;
        if (values[i] < -1.0) values[i] = -1.0;
    }
}

static void initialise_magnetisations(double target_M) {
    /* Feasible, slightly perturbed initial point. */
    const double base = target_M / (double)N;
    for (int i = 0; i < N; ++i)
        m[i] = base + 0.02 * (2.0 * urand01() - 1.0);
    project_box_sum(m, target_M);
}

static void run_mean_field(int k, double gamma0, double T0, double Tmin, int sweeps) {
    const double target_M = 2.0 * (double)k - (double)N;
    initialise_magnetisations(target_M);

    for (int t = 0; t < sweeps; ++t) {
        const double fraction = (sweeps <= 1) ? 1.0 : (double)t / (double)(sweeps - 1);
        /* Smooth schedules; temperature never reaches zero. */
        const double gamma = gamma0 * (1.0 - fraction);
        const double temperature = T0 * pow(Tmin / T0, fraction);

        /* Compute the interaction field from the previous sweep (Jacobi update). */
        for (int i = 0; i < N; ++i) {
            double sum = h[i];
            for (int j = 0; j < N; ++j)
                if (i != j) sum += J[i][j] * m[j];
            local_field[i] = sum;
        }

        const double lambda = solve_lambda(target_M, gamma, temperature);
        for (int i = 0; i < N; ++i) {
            const double z = local_field[i] - lambda;
            const double candidate = response(z, gamma, temperature);
            m_new[i] = (1.0 - DAMPING) * m[i] + DAMPING * candidate;
        }

        /* Keep both the box bounds and exact magnetisation constraint. */
        project_box_sum(m_new, target_M);
        for (int i = 0; i < N; ++i) m[i] = m_new[i];
    }
}

/* Select exactly k largest m_i as +1; all other spins are -1. */
static void project_to_exact_cardinality(int k, int spins[N]) {
    int order[N];
    for (int i = 0; i < N; ++i) order[i] = i;
    for (int i = 1; i < N; ++i) {
        const int key = order[i];
        int j = i - 1;
        while (j >= 0 && m[order[j]] < m[key]) {
            order[j + 1] = order[j];
            --j;
        }
        order[j + 1] = key;
    }
    for (int i = 0; i < N; ++i) spins[i] = -1;
    for (int i = 0; i < k; ++i) spins[order[i]] = +1;
}

static double ising_energy(const int spins[N]) {
    double energy = 0.0;
    for (int i = 0; i < N; ++i) {
        energy -= h[i] * (double)spins[i];
        for (int j = i + 1; j < N; ++j)
            energy -= J[i][j] * (double)spins[i] * (double)spins[j];
    }
    return energy;
}

int main(int argc, char **argv) {
    int k = DEFAULT_K;
    double gamma0 = DEFAULT_GAMMA0;
    double T0 = DEFAULT_T0;
    double Tmin = DEFAULT_TMIN;
    int sweeps = DEFAULT_SWEEPS;
    int nrun = DEFAULT_NRUN;
    unsigned int seed = (unsigned int)time(NULL);

    if (argc > 1) k = atoi(argv[1]);
    if (argc > 2) gamma0 = atof(argv[2]);
    if (argc > 3) T0 = atof(argv[3]);
    if (argc > 4) Tmin = atof(argv[4]);
    if (argc > 5) sweeps = atoi(argv[5]);
    if (argc > 6) nrun = atoi(argv[6]);
    if (argc > 7) seed = (unsigned int)strtoul(argv[7], NULL, 10);

    if (k < 1 || k >= N || gamma0 < 0.0 || T0 <= 0.0 || Tmin <= 0.0 ||
        Tmin > T0 || sweeps < 1 || nrun < 1) {
        fprintf(stderr,
                "Usage: %s [k(1..%d)] [gamma0>=0] [T0>0] [Tmin in (0,T0]] [sweeps>=1] [nrun>=1] [seed]\n",
                argv[0], N - 1);
        return EXIT_FAILURE;
    }

    srand(seed);
    read_inputs();

    FILE *out = fopen(OUTFILE, "w");
    if (!out) {
        fprintf(stderr, "Cannot open %s: %s\n", OUTFILE, strerror(errno));
        return EXIT_FAILURE;
    }

    const clock_t start = clock();
    int cardinality_errors = 0;
    double best_energy = INFINITY;

    for (int run = 0; run < nrun; ++run) {
        int spins[N];
        run_mean_field(k, gamma0, T0, Tmin, sweeps);
        project_to_exact_cardinality(k, spins);

        int selected = 0;
        for (int i = 0; i < N; ++i) if (spins[i] == +1) ++selected;
        if (selected != k) ++cardinality_errors;

        const double energy = ising_energy(spins);
        fprintf(out, "%.12f", energy);
        for (int i = 0; i < N; ++i) fprintf(out, " %d", spins[i]);
        fputc('\n', out);

        if (energy < best_energy) {
            best_energy = energy;
        }
    }

    const double seconds = (double)(clock() - start) / CLOCKS_PER_SEC;
    fprintf(out, "# runs=%d k=%d sweeps=%d gamma0=%.8g T0=%.8g Tmin=%.8g seed=%u\n",
            nrun, k, sweeps, gamma0, T0, Tmin, seed);
    fprintf(out, "# cardinality_errors=%d best_energy=%.12f elapsed_seconds=%.3f\n",
            cardinality_errors, best_energy, seconds);
    fclose(out);

    printf("Completed %d runs. Every run targets exactly k=%d; cardinality errors=%d.\n",
           nrun, k, cardinality_errors);
    printf("Best energy: %.12f | output: %s | elapsed: %.3f s | seed: %u\n",
           best_energy, OUTFILE, seconds, seed);
    return cardinality_errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
