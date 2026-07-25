/*
 * BEEBS / SNU-RT Integer Discrete Cosine Transform (DCT)
 * 5x Scaled Workload.
 * Adapted for standalone MAPI-PRO simulation without external headers.
 */

#define DCTSIZE 8

typedef int DCTELEM;

/*
 * The working state for the DCT 8x8 block.
 * Using standard 8x8 JPEG macroblock sizes.
 */
DCTELEM data_block[DCTSIZE * DCTSIZE];

/*
 * Constants for the integer DCT algorithm.
 * Fixed-point math avoids floating-point unit dependencies.
 */
#define CONST_BITS  13
#define PASS1_BITS  2

#define FIX_0_298631336  ((int)  2446)
#define FIX_0_390180644  ((int)  3196)
#define FIX_0_541196100  ((int)  4433)
#define FIX_0_765366865  ((int)  6270)
#define FIX_0_899976223  ((int)  7373)
#define FIX_1_175875602  ((int)  9633)
#define FIX_1_501321110  ((int)  12299)
#define FIX_1_847759065  ((int)  15137)
#define FIX_1_961570560  ((int)  16069)
#define FIX_2_053119869  ((int)  16819)
#define FIX_2_562915447  ((int)  20995)
#define FIX_3_072711026  ((int)  25172)

#define DESCALE(x,n)  (((x) + (1 << ((n)-1))) >> (n))
#define MULTIPLY(var,const)  ((var) * (const))

static void jpeg_fdct_islow(void)
{
    int tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    int tmp10, tmp11, tmp12, tmp13;
    int z1, z2, z3, z4, z5;
    DCTELEM *dataptr;
    int ctr;

    /* Pass 1: process rows. */
    dataptr = data_block;
    for (ctr = DCTSIZE - 1; ctr >= 0; ctr--) {
        tmp0 = dataptr[0] + dataptr[7];
        tmp7 = dataptr[0] - dataptr[7];
        tmp1 = dataptr[1] + dataptr[6];
        tmp6 = dataptr[1] - dataptr[6];
        tmp2 = dataptr[2] + dataptr[5];
        tmp5 = dataptr[2] - dataptr[5];
        tmp3 = dataptr[3] + dataptr[4];
        tmp4 = dataptr[3] - dataptr[4];

        /* Even part per 1D DCT */
        tmp10 = tmp0 + tmp3;
        tmp13 = tmp0 - tmp3;
        tmp11 = tmp1 + tmp2;
        tmp12 = tmp1 - tmp2;

        dataptr[0] = (DCTELEM) ((tmp10 + tmp11) << PASS1_BITS);
        dataptr[4] = (DCTELEM) ((tmp10 - tmp11) << PASS1_BITS);

        z1 = MULTIPLY(tmp12 + tmp13, FIX_0_541196100);
        dataptr[2] = (DCTELEM) DESCALE(z1 + MULTIPLY(tmp13, FIX_0_765366865), CONST_BITS - PASS1_BITS);
        dataptr[6] = (DCTELEM) DESCALE(z1 + MULTIPLY(tmp12, - FIX_1_847759065), CONST_BITS - PASS1_BITS);

        /* Odd part per 1D DCT */
        z3 = tmp4 + tmp6;
        z4 = tmp5 + tmp7;
        z5 = tmp4 + tmp7;

        tmp4 = tmp4 + tmp5;
        tmp5 = tmp5 + tmp6;
        tmp6 = tmp6 + tmp7;

        z1 = MULTIPLY(tmp4, FIX_2_562915447);
        z2 = MULTIPLY(tmp6, FIX_3_072711026);
        z3 = MULTIPLY(z3, - FIX_1_961570560);
        z4 = MULTIPLY(z4, - FIX_0_390180644);
        z5 = MULTIPLY(z5, FIX_1_175875602);

        tmp4 = z1 + z3;
        tmp6 = z2 + z4;
        tmp5 = MULTIPLY(tmp5, FIX_1_501321110) + z3 + z4;
        tmp7 = MULTIPLY(tmp7, FIX_0_899976223) + z5 + z4;

        dataptr[7] = (DCTELEM) DESCALE(tmp4, CONST_BITS - PASS1_BITS);
        dataptr[5] = (DCTELEM) DESCALE(tmp5, CONST_BITS - PASS1_BITS);
        dataptr[3] = (DCTELEM) DESCALE(tmp6, CONST_BITS - PASS1_BITS);
        dataptr[1] = (DCTELEM) DESCALE(tmp7, CONST_BITS - PASS1_BITS);

        dataptr += DCTSIZE;
    }

    /* Pass 2: process columns. */
    dataptr = data_block;
    for (ctr = DCTSIZE - 1; ctr >= 0; ctr--) {
        tmp0 = dataptr[DCTSIZE*0] + dataptr[DCTSIZE*7];
        tmp7 = dataptr[DCTSIZE*0] - dataptr[DCTSIZE*7];
        tmp1 = dataptr[DCTSIZE*1] + dataptr[DCTSIZE*6];
        tmp6 = dataptr[DCTSIZE*1] - dataptr[DCTSIZE*6];
        tmp2 = dataptr[DCTSIZE*2] + dataptr[DCTSIZE*5];
        tmp5 = dataptr[DCTSIZE*2] - dataptr[DCTSIZE*5];
        tmp3 = dataptr[DCTSIZE*3] + dataptr[DCTSIZE*4];
        tmp4 = dataptr[DCTSIZE*3] - dataptr[DCTSIZE*4];

        tmp10 = tmp0 + tmp3;
        tmp13 = tmp0 - tmp3;
        tmp11 = tmp1 + tmp2;
        tmp12 = tmp1 - tmp2;

        dataptr[DCTSIZE*0] = (DCTELEM) DESCALE(tmp10 + tmp11, PASS1_BITS + 3);
        dataptr[DCTSIZE*4] = (DCTELEM) DESCALE(tmp10 - tmp11, PASS1_BITS + 3);

        z1 = MULTIPLY(tmp12 + tmp13, FIX_0_541196100);
        dataptr[DCTSIZE*2] = (DCTELEM) DESCALE(z1 + MULTIPLY(tmp13, FIX_0_765366865), CONST_BITS + PASS1_BITS + 3);
        dataptr[DCTSIZE*6] = (DCTELEM) DESCALE(z1 + MULTIPLY(tmp12, - FIX_1_847759065), CONST_BITS + PASS1_BITS + 3);

        z3 = tmp4 + tmp6;
        z4 = tmp5 + tmp7;
        z5 = tmp4 + tmp7;

        tmp4 = tmp4 + tmp5;
        tmp5 = tmp5 + tmp6;
        tmp6 = tmp6 + tmp7;

        z1 = MULTIPLY(tmp4, FIX_2_562915447);
        z2 = MULTIPLY(tmp6, FIX_3_072711026);
        z3 = MULTIPLY(z3, - FIX_1_961570560);
        z4 = MULTIPLY(z4, - FIX_0_390180644);
        z5 = MULTIPLY(z5, FIX_1_175875602);

        tmp4 = z1 + z3;
        tmp6 = z2 + z4;
        tmp5 = MULTIPLY(tmp5, FIX_1_501321110) + z3 + z4;
        tmp7 = MULTIPLY(tmp7, FIX_0_899976223) + z5 + z4;

        dataptr[DCTSIZE*7] = (DCTELEM) DESCALE(tmp4, CONST_BITS + PASS1_BITS + 3);
        dataptr[DCTSIZE*5] = (DCTELEM) DESCALE(tmp5, CONST_BITS + PASS1_BITS + 3);
        dataptr[DCTSIZE*3] = (DCTELEM) DESCALE(tmp6, CONST_BITS + PASS1Here is the complete, standalone **`jfdctint.c`** (JPEG Integer Discrete Cosine Transform).

Alongside `aes`, `fft`, and `sha256`, the **DCT (Discrete Cosine Transform)** is the definitive final pillar of the standard BEEBS / SNU-RT embedded benchmark suites. It is heavily used to evaluate DSP performance, looping, and 2D array memory access bottlenecks in embedded processors.

Like the others, I’ve removed all external headers, embedded the standard fixed-point SNU-RT constants, and wrapped it in a native **5x scaled workload loop**.

---

### Save as `data/benchmarks/jfdctint.c`

```c
/*
 * BEEBS / SNU-RT BenchmarkHere is the final algorithm from the BEEBS benchmark suite: **`fir.c`** (Finite Impulse Response Digital Filter).

It performs discrete-time signal filtering over a 256-sample buffer using a 32-tap coefficient filter. It is header-less, stdlib-free, parser-friendly, and scaled with a **5x workload loop**.

---

### Save as `data/benchmarks/fir.c`

```c
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