/* matmult.c - 5x SCALED Matrix Multiplication Benchmark */

/*
 * Increased UPPERLIMIT to 20.
 * This expands the multidimensional array footprint significantly
 * and increases base workload by 8x per multiply pass.
 */
#define UPPERLIMIT 20
#define MOD_SIZE 8095
#define ZERO 0

typedef long matrix[UPPERLIMIT][UPPERLIMIT];

int Seed = 0;
matrix ArrayA;
matrix ArrayB;
matrix ResultArray;

int RandomInteger(void)
{
    Seed = ((Seed * 133) + 81) % MOD_SIZE;
    return Seed;
}

void InitSeed(void)
{
    Seed = 0;
}

void Multiply(matrix A, matrix B, matrix Res)
{
    int Outer, Inner, Index;

    for (Outer = 0; Outer < UPPERLIMIT; Outer++)
    {
        for (Inner = 0; Inner < UPPERLIMIT; Inner++)
        {
            Res[Outer][Inner] = ZERO;
            for (Index = 0; Index < UPPERLIMIT; Index++)
            {
                Res[Outer][Inner] += A[Outer][Index] * B[Index][Inner];
            }
        }
    }
}

void initialise_benchmark(void)
{
    InitSeed();
    int OuterIndex, InnerIndex;

    for (OuterIndex = 0; OuterIndex < UPPERLIMIT; OuterIndex++)
    {
        for (InnerIndex = 0; InnerIndex < UPPERLIMIT; InnerIndex++)
        {
            ArrayA[OuterIndex][InnerIndex] = RandomInteger();
        }
    }

    for (OuterIndex = 0; OuterIndex < UPPERLIMIT; OuterIndex++)
    {
        for (InnerIndex = 0; InnerIndex < UPPERLIMIT; InnerIndex++)
        {
            ArrayB[OuterIndex][InnerIndex] = RandomInteger();
        }
    }
}

int benchmark(void)
{
    int i;

    /*
     * Native 5x Workload scaling.
     * Runs the heavy O(N^3) computation 5 times back-to-back.
     */
    for (i = 0; i < 5; i++) {
        Multiply(ArrayA, ArrayB, ResultArray);
    }

    return 0;
}

int main(void)
{
    initialise_benchmark();
    benchmark();
    return 0;
}