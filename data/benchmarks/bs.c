/*
 * BEEBS Binary Search (bs) Benchmark (5x Scaled Workload)
 * Original code based on SNU-RT / BEEBS benchmark suite.
 * Adapted for standalone MAPI-PRO simulation without external headers.
 */

#define DATA_SIZE 128

typedef struct {
    int key;
    int value;
} Record;

static Record data_array[DATA_SIZE];
static int search_targets[16] = {
    2, 15, 34, 58, 72, 89, 102, 115, 130, 148, 160, 185, 200, 220, 240, 300
};

void initialise_benchmark(void)
{
    int i;
    for (i = 0; i < DATA_SIZE; i++) {
        data_array[i].key = i * 2; // Monotonically sorted keys: 0, 2, 4, ... 254
        data_array[i].value = i * 10;
    }
}

int binary_search(int target)
{
    int low = 0;
    int high = DATA_SIZE - 1;
    int mid;

    while (low <= high) {
        mid = low + ((high - low) >> 1);

        if (data_array[mid].key == target) {
            return data_array[mid].value;
        }
        if (data_array[mid].key < target) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return -1;
}

int benchmark(void)
{
    int pass;
    int i;
    int found_count = 0;

    /*
     * Native 5x Workload scaling.
     * Executes repeated binary search operations over varying target keys.
     */
    for (pass = 0; pass < 5; pass++) {
        for (i = 0; i < 16; i++) {
            if (binary_search(search_targets[i]) != -1) {
                found_count++;
            }
        }
    }

    return found_count;
}

int main(void)
{
    initialise_benchmark();
    benchmark();
    return 0;
}