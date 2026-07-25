/*
 * BEEBS FIR Filter Benchmark (5x Scaled Workload)
 * Finite Impulse Response (FIR) digital signal filtering.
 * Adapted for standalone MAPI-PRO simulation without external headers.
 */

#define FIR_NTAPS 32
#define FIR_NSAMPLES 256

int fir_input[FIR_NSAMPLES];
int fir_output[FIR_NSAMPLES];

// 32-tap fixed-point filter coefficients
static const int fir_coeffs[FIR_NTAPS] = {
    -1, -2, -3, -4, 0, 8, 20, 35,
    50, 62, 68, 62, 50, 35, 20, 8,
    0, -4, -3, -2, -1, 2, 5, 8,
    10, 8, 5, 2, -1, -2, -3, -4
};

static void fir_filter(void)
{
    int i, j;
    int sum;

    for (i = 0; i < FIR_NSAMPLES; i++) {
        sum = 0;
        for (j = 0; j < FIR_NTAPS; j++) {
            if (i - j >= 0) {
                sum += fir_input[i - j] * fir_coeffs[j];
            }
        }
        fir_output[i] = sum >> 8;
    }
}

void initialise_benchmark(void)
{
    int i;
    for (i = 0; i < FIR_NSAMPLES; i++) {
        fir_input[i] = ((i * 37 + 11) % 256) - 128;
        fir_output[i] = 0;
    }
}

int benchmark(void)
{
    int pass;

    /*
     * Native 5x Workload scaling.
     * Executes 5 complete filtering passes across the input signal buffer.
     */
    for (pass = 0; pass < 5; pass++) {
        fir_filter();
    }

    return 0;
}

int main(void)
{
    initialise_benchmark();
    benchmark();
    return 0;
}