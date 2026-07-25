/*
 * BEEBS Count (cnt) Benchmark (5x Scaled Workload)
 * Original code based on SNU-RT / MRTC / BEEBS benchmark suite.
 * Adapted for standalone MAPI-PRO simulation without external headers.
 */

#define MAXSIZE 64

// 2D Volatile state array mapped to FRAM
static int Array[MAXSIZE][MAXSIZE];

void initialise_benchmark(void)
{
    int InitSeed = 0;
    int i, j;

    // Generate deterministic pseudo-random values (mix of positive and negative)
    for (i = 0; i < MAXSIZE; i++) {
        for (j = 0; j < MAXSIZE; j++) {
            InitSeed = (InitSeed * 13 + 17) % 300;
            Array[i][j] = InitSeed - 150;
        }
    }
}

int count_non_negative(void)
{
    int i, j;
    int cnt = 0;

    for (i = 0; i < MAXSIZE; i++) {
        for (j = 0; j < MAXSIZE; j++) {
            if (Array[i][j] >= 0) {
                cnt++;
            }
        }
    }
    return cnt;
}

int benchmark(void)
{
    int pass;
    int total_cnt = 0;

    /*
     * Native 5x Workload scaling.
     * Executes repeated matrix traversal and counting operations.
     */
    for (pass = 0; pass < 5; pass++) {
        total_cnt += count_non_negative();
    }

    return total_cnt;
}

int main(void)
{
    initialise_benchmark();
    benchmark();
    return 0;
}