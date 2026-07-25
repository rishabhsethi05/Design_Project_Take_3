/*
 * BEEBS / SNU-RT FFT Benchmark (5x Scaled Workload)
 * Fast Fourier Transform (128-point Radix-2 Cooley-Tukey).
 * Adapted for standalone MAPI-PRO simulation without external headers.
 */

#define FFT_SIZE 128
#define LOG_N 7

float RealData[FFT_SIZE];
float ImagData[FFT_SIZE];

/* Fixed sine table for 128 points to avoid <math.h> dependencies */
static const float SinTable[64] = {
    0.000000f, 0.049068f, 0.098017f, 0.146730f, 0.195090f, 0.242980f, 0.290285f, 0.336890f,
    0.382683f, 0.427555f, 0.471397f, 0.514103f, 0.555570f, 0.595699f, 0.634393f, 0.671559f,
    0.707107f, 0.740951f, 0.773010f, 0.803208f, 0.831470f, 0.857729f, 0.881921f, 0.903989f,
    0.923880f, 0.941544f, 0.956940f, 0.970031f, 0.980785f, 0.989177f, 0.995185f, 0.998795f,
    1.000000f, 0.998795f, 0.995185f, 0.989177f, 0.980785f, 0.970031f, 0.956940f, 0.941544f,
    0.923880f, 0.903989f, 0.881921f, 0.857729f, 0.831470f, 0.803208f, 0.773010f, 0.740951f,
    0.707107f, 0.671559f, 0.634393f, 0.595699f, 0.555570f, 0.514103f, 0.471397f, 0.427555f,
    0.382683f, 0.336890f, 0.290285f, 0.242980f, 0.195090f, 0.146730f, 0.098017f, 0.049068f
};

static float get_sin(int idx)
{
    idx = idx % 128;
    if (idx < 0) idx += 128;
    if (idx < 64) return SinTable[idx];
    return -SinTable[idx - 64];
}

static float get_cos(int idx)
{
    return get_sin(idx + 32);
}

static unsigned int reverse_bits(unsigned int val, int bits)
{
    unsigned int rev = 0;
    int i;
    for (i = 0; i < bits; i++) {
        rev = (rev << 1) | (val & 1);
        val >>= 1;
    }
    return rev;
}

static void fft_compute(void)
{
    int i, j;
    int n = FFT_SIZE;

    /* Bit-reversal permutation */
    for (i = 0; i < n; i++) {
        j = reverse_bits(i, LOG_N);
        if (j > i) {
            float tempR = RealData[i];
            float tempI = ImagData[i];
            RealData[i] = RealData[j];
            ImagData[i] = ImagData[j];
            RealData[j] = tempR;
            ImagData[j] = tempI;
        }
    }

    /* Cooley-Tukey Radix-2 Butterfly Computation */
    int len;
    for (len = 2; len <= n; len <<= 1) {
        int half_len = len >> 1;
        int step = n / len;
        for (i = 0; i < n; i += len) {
            for (j = 0; j < half_len; j++) {
                int k_idx = j * step;
                float uR = RealData[i + j];
                float uI = ImagData[i + j];
                float cos_val = get_cos(k_idx);
                float sin_val = get_sin(k_idx);

                float vR = RealData[i + j + half_len] * cos_val + ImagData[i + j + half_len] * sin_val;
                float vI = ImagData[i + j + half_len] * cos_val - RealData[i + j + half_len] * sin_val;

                RealData[i + j] = uR + vR;
                ImagData[i + j] = uI + vI;
                RealData[i + j + half_len] = uR - vR;
                ImagData[i + j + half_len] = uI - vI;
            }
        }
    }
}

void initialise_benchmark(void)
{
    int i;
    for (i = 0; i < FFT_SIZE; i++) {
        RealData[i] = (float)(i % 17) - 8.0f;
        ImagData[i] = 0.0f;
    }
}

int benchmark(void)
{
    int pass;

    /*
     * Native 5x Workload scaling.
     * Performs 5 full 128-point Fast Fourier Transforms sequentially.
     */
    for (pass = 0; pass < 5; pass++) {
        fft_compute();
    }

    return 0;
}

int main(void)
{
    initialise_benchmark();
    benchmark();
    return 0;
}