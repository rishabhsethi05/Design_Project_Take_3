/*
 * BEEBS Select Benchmark (5x Scaled Workload)
 * Original code based on SNU-RT / MRTC / BEEBS benchmark suite.
 * Adapted for standalone MAPI-PRO simulation without external headers.
 */

#define SIZE 128

// Volatile state arrays mapped to FRAM
static int arr[SIZE];
static int backup_arr[SIZE];

void initialise_benchmark(void)
{
    int i;
    // Generate deterministic pseudo-random values
    for (i = 0; i < SIZE; i++) {
        backup_arr[i] = (i * 73) % 256;
    }
}

int quick_select(int k)
{
    int i, j, left, right;
    int pivot, temp;

    left = 0;
    right = SIZE - 1;

    while (left < right) {
        pivot = arr[k];
        i = left;
        j = right;

        do {
            while (arr[i] < pivot) i++;
            while (pivot < arr[j]) j--;

            if (i <= j) {
                temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
                i++;
                j--;
            }
        } while (i <= j);

        if (j < k) left = i;
        if (k < i) right = j;
    }
    return arr[k];
}

int benchmark(void)
{
    int pass, i;
    int result = 0;

    /*
     * Native 5x Workload scaling.
     * Executes repeated Quick Select operations over the array.
     */
    for (pass = 0; pass < 5; pass++) {
        // Restore array state for each pass
        for (i = 0; i < SIZE; i++) {
            arr[i] = backup_arr[i];
        }

        // Search for 5th, 25th, 75th, and 100th smallest elements
        result += quick_select(5);
        result += quick_select(25);
        result += quick_select(75);
        result += quick_select(100);
    }

    return result;
}

int main(void)
{
    initialise_benchmark();
    benchmark();
    return 0;
}